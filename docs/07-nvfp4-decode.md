# NVFP4 decode: W4A16, W4A4 и fused MLP

## Что реализовано

В `qwc-cuda` появился первый настоящий projection dataplane:

* `nvfp4::Linear` загружает `weight_packed` и `weight_scale` прямо из
  compressed-tensors checkpoint;
* `forward_w4a16` считает NVFP4-веса × BF16-активации для batch 1..4;
* `swiglu_w4a16` объединяет `gate_proj + up_proj + SiLU × mul` в один kernel;
* собственный BF16→NVFP4 quantizer сразу пишет packed E2M1 и SM120 128×4
  scale-layout без промежуточной row-major копии;
* `forward_w4a4_quantized` запускает block-scaled tensor-core MMA для batch
  3..32 с BF16-выходом;
* `select_decode_kernel` выбирает W4A16/W4A4 по batch и форме проекции;
* есть CPU-oracle форматов E2M1/E4M3, синтетический GPU-test и проверка на
  настоящих тензорах QUASAR.

Это пока decode-path малого batch, не полный token loop и не prefill.

## Формат и масштабы

Checkpoint хранит:

```text
weight_packed        u8      [out, in/2]   два E2M1 в байте
weight_scale         e4m3    [out, in/16]  локальный scale
weight_global_scale  f32     [1]           глобальный делитель
```

Compressed-tensors хранит глобальный scale как делитель. Поэтому для W4A16:

```text
W[row, k] = E2M1(weight_packed[row, k])
          * E4M3(weight_scale[row, k/16])
          / weight_global_scale

Y = X_bf16 * W^T
```

`input_global_scale` нужен только для W4A4 activation quantization и в W4A16
намеренно не используется. Для каждой группы из 16 активаций:

```text
SFA = E4M3(input_global_scale * max(abs(X_group)) / 6)
A4  = E2M1_RNE(X_group * input_global_scale / SFA)

Y = alpha * MMA(A4 * SFA, W4 * SFW)
alpha = 1 / (input_global_scale * weight_global_scale)
```

Формула и трактовка serialized globals как делителей совпадают с обработкой
NVFP4 в [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py).

4-битные веса ни в одном тракте не репакуются: checkpoint `[N,K]` row-major
физически совпадает с требуемым GEMM `B[K,N]` column-major. Scale-матрица один
раз при загрузке переставляется блоками 128×4 и остаётся единственной копией:
и W4A4, и самописный W4A16 читают этот layout. Если бы raw и swizzled scales
хранились одновременно, на всех quantized-параметрах модели это стоило бы ещё
1.52 GB VRAM.

## Раскладка kernel

Основной tile считает 16 выходных строк блоком из 128 потоков:

```text
threadIdx.x = 8 K-потоков на строку
threadIdx.y = 16 выходных строк
K-tile      = 256
X tile      = BF16 [batch, 256] в shared memory
W fragment  = 128 bit на поток
```

Activation tile загружается в shared один раз на 16 строк. Фрагмент веса
читается один раз и используется сразу для всех последовательностей batch.
Batch 1, 2, 3 и 4 — отдельные compile-time kernels: B=1 не платит регистрами
и ветвлениями за B=4.

Для узких проекций (`out < 8192`) используется другая раскладка: 32 K-потока
на строку и K-tile 512. Это удваивает число независимых варпов и вдвое сокращает
число барьеров на длинной `down_proj [5120, 17408]`.

Fused MLP читает две матрицы, но загружает X только один раз и сразу пишет:

```text
intermediate = silu(gate_proj(X)) * up_proj(X)
```

Отдельных буферов gate/up и отдельного elementwise kernel нет.

W4A4 использует SM120 block-scaled MMA:

```text
tile             128 × 128 × 256
schedule         warp-specialized ping-pong
A                packed E2M1 [batch,K], row-major
B                checkpoint packed E2M1 [N,K], как column-major [K,N]
SFA/SFB          E4M3, CUTLASS 128×4 swizzle
D                BF16 [batch,N]
workspace        0 bytes
```

Стоковый CUTLASS example использует K=128 cooperative. Прямой перебор на RTX
5090 показал, что K=256 ping-pong лучше; N=64 и K=256 cooperative отброшены
после отрицательных A/B-замеров.

## Корректность

Синтетический тест проверяет batch 1..4, обе аппаратные раскладки и неполный
последний tile. Отдельный пример загрузил настоящие тензоры
`layers.0.mlp.gate_proj/up_proj` из `Qwen3.8-27B-QUASAR-NVFP4`.

```text
gate shape                         17408 × 5120
checkpoint weight_global_scale     6371.5557
projection max scaled error        0.003859
fused SwiGLU max scaled error       0.003731
W4A4 batch=4 max scaled error       0.003887
CUTLASS workspace                   0 B
```

Ошибка включает BF16-округление выхода. Bit layout, порядок nibble и оба уровня
scale подтверждены реальным checkpoint, а не только синтетикой.

## RTX 5090: streaming benchmark

Команда:

```bash
cargo run --release -p qwc-cuda --bin nvfp4bench
```

Бенчмарк циклически проходит четыре разные матрицы, поэтому рабочее множество
больше 96 MiB L2. Число — медиана семи прогретых серий. На загруженном desktop
GPU частоты и конкурирующие графические процессы дают заметный разброс, поэтому
для решений важны прямые A/B-замеры в одной серии.

Fused MLP против двух отдельных projection launches (без отдельного SiLU в
baseline, то есть сравнение консервативно):

| batch | две проекции | fused SwiGLU | ускорение |
|---:|---:|---:|---:|
| 1 | 103.16 us | 99.19 us | 1.04× |
| 2 | 105.27 us | 99.61 us | 1.06× |
| 3 | 152.16 us | 100.38 us | 1.52× |
| 4 | 192.31 us | 116.86 us | 1.65× |

Главный результат — один и тот же вес обслуживает несколько последовательностей.
На ранних изолированных сериях отдельные gate/q+gate projections для B=1–2
достигали 84–88% измеренного DRAM-потолка, а две последовательности часто
считались практически за время одной. Медианный benchmark намеренно не скрывает
desktop-contention и печатает текущую долю потолка при каждом запуске.

Для сравнения проверочный CUTLASS `91_fp4_gemv` на четырёх разных gate/up
матрицах показал 0.135 ms на весь streaming-вызов и 1384 GiB/s. Это сильный
batch=1 oracle, но его штатный batch дублирует матрицу A, scale layout swizzled,
а выход снова FP4. Он не заменяет shared-weight server path.

### W4A4 crossover

Quantizer занимает примерно 2.3 us для batch 3..32. Ниже прямой A/B-замер
K=256 ping-pong; W4A4-числа уже включают quantizer:

| форма | batch | W4A16 | W4A4 total | W4A4 / A16 |
|---|---:|---:|---:|---:|
| gate/up 17408×5120 | 4 | 91.66 us | 75.60 us | 1.21× |
| q+gate 12288×5120 | 4 | 43.88 us | 41.10 us | 1.07× |
| down 5120×17408 | 4 | 50.67 us | 66.60 us | 0.76× |

При batch=32 один W4A4 projection занял 69.34 us для gate/up, 63.62 us для
down и 36.05 us для q+gate. Desktop-конкуренция заметно меняет абсолютные числа,
поэтому dispatcher опирается на относительный A/B-результат одной серии.

## Что ещё не доказано

* W4A16 убирает ошибку квантизации активаций, но влияние на качество всей QAT
  модели надо подтвердить perplexity/task-eval; «больше разрядность» не считается
  автоматическим доказательством качества.
* Fused MLP пока не включает `down_proj`, residual и RMSNorm.
* W4A4 MLP пока запускает gate/up как отдельные GEMM; нужен общий persistent
  mainloop или fused layout, чтобы одна волна CTA стримила обе матрицы.
* Влияние W4A4 prefill/decode на perplexity относительно эталонного runtime
  ещё нужно измерить отдельно.

## Целевая диспетчеризация

```text
batch 1..3   W4A16 custom GEMV / fused MLP  — качество, низкая latency
batch 4      N>K: W4A4; N<=K: W4A16         — измеренная граница по форме
batch 5..32  W4A4 SM120 MMA                 — один проход по весам
prefill      W4A4 SM120 GEMM, M=128          — chunked causal scan
```

Граница не зашивается навсегда: финальный autotune выбирает её отдельно для
каждой формы по goodput и ITL, а quality profile сможет принудительно оставлять
W4A16 дольше.
