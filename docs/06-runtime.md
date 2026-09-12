# Runtime: двухресурсный кэш и continuous batching

## Зачем отдельный runtime

У Qwen3.8 одна последовательность одновременно занимает два разных ресурса:

* фиксированный state-слот для 48 Gated DeltaNet слоёв;
* переменное число paged KV-блоков для 16 full-attention слоёв.

Поэтому одного счётчика KV недостаточно. Запрос принимается только тогда, когда
есть и state-слот, и все KV-блоки prompt. `CacheManager` выполняет эту проверку
транзакционно: при отказе ни один ресурс не расходуется. Scheduler дополнительно
коммитит KV-ёмкость на `prompt + max_new_tokens`, пока нет preemption: иначе
несколько запросов могли бы вместе заполнить пул на границе страниц и навсегда
застрять перед следующим decode-токеном.

`CacheManager::from_budget` статически делит доступную VRAM под заданный
`target_context`. Число state-слотов равно `Budget::max_concurrency`, остаток
отдаётся KV-пулу. Если реальные запросы заметно короче target, раньше
заканчиваются state-слоты; если длиннее — KV-блоки.

## Состояния запроса

```text
waiting
  -> prefill-ready <-> prefill-in-flight
  -> decode-ready  <-> decode-in-flight
  -> completed / cancelled -> release(state + KV)
```

Одновременно существует не больше одного in-flight batch. Это не ограничение
GPU: внутри batch может быть до `max_num_seqs` последовательностей. Ограничение
нужно на границе CPU/GPU, чтобы подтверждение и откат всегда относились к
единственному известному плану.

Жизненный цикл executor:

1. `submit` кладёт запрос в FIFO, ещё не занимая VRAM.
2. `next_batch` принимает помещающиеся запросы и возвращает смешанный `Batch`.
3. Executor выполняет `decode` и `prefill` части batch.
4. При успехе вызывает `complete_batch(stopped)`, при ошибке до исполнения —
   `abort_batch()`.

## Политика планирования

1. Готовый decode имеет приоритет: это защищает inter-token latency уже
   запущенных запросов.
2. Оставшиеся sequence slots и token budget заполняются chunked prefill.
3. Decode обслуживается round-robin.
4. Admission идёт в порядке FIFO, но длинный prompt, временно не помещающийся
   в свободный KV, не блокирует меньший запрос за ним. Отсутствие state-слота
   останавливает admission целиком, потому что следующий запрос тоже не
   сможет пройти.

`max_num_batched_tokens` общий: один decode стоит один токен, prefill — размер
чанка. Поэтому scheduler может в одном шаге поддержать decode активных
последовательностей и использовать остаток GPU-работы для нового prompt.

## Резервация до GPU

Перед добавлением последовательности в decode batch scheduler вызывает
`CacheManager::append_token`. Если следующий токен пересекает границу страницы,
новый KV-блок выделяется **до** запуска кернела. OOM между планированием и
исполнением невозможен.

Коммит lifetime-ёмкости логический: реальные страницы по-прежнему выделяются
лениво. Поэтому block tables содержат только уже используемые страницы, но
admission не обещает свободный остаток другим запросам. Консервативность можно
будет снять одновременно с появлением preemption, а не ценой дедлока.

Если batch не был исполнен, `abort_batch` вызывает `rollback_token`: счётчик
длины возвращается назад, а лишняя KV-страница освобождается. Prefill откатить
проще — весь prompt-кэш резервируется при admission, а подтверждённое смещение
меняется только в `complete_batch`.

Завершение по лимиту токенов, EOS/stop condition и явная отмена освобождают оба
ресурса. Отмена in-flight последовательности запрещена: сначала необходимо
подтвердить или откатить batch, чтобы GPU не получил dangling block table.

## Batch metadata для GPU

`BatchLayout` переводит scheduler batch и текущее состояние кэша в плоскую
structure-of-arrays раскладку на `u32`. Decode-записи идут первыми, затем
prefill-чанки. Для каждой последовательности передаются:

* `SeqId` и номер state-слота;
* начальная позиция и длина видимого контекста;
* CSR offset входных токенов;
* CSR offset и физические ID KV-блоков.

Для decode длина уже включает зарезервированный следующий токен. Для prefill
она заканчивается на границе текущего чанка, даже если admission заранее
выделил страницы под весь prompt. Такая раскладка не содержит Rust padding и
копируется в преаллоцированные device buffers фиксированного адреса. CUDA graph
читает новые значения из тех же адресов на каждом запуске.

## Проверка

```bash
cargo test -p qwc-runtime
cargo clippy -p qwc-runtime --all-targets -- -D warnings \
  -A clippy::manual-is-multiple-of
```

CPU-тесты покрывают оба вида дефицита, ленивое расширение KV, rollback на
границе страницы, chunked prefill, mixed batch, first-fit admission, лимит
генерации, stop condition, безопасную отмену и CSR-раскладку GPU metadata.

`qwc-engine::Executor::execute_layout` принимает эту раскладку напрямую:
decode использует scheduler-selected state slots и физические KV pages, а
prefill-чанки обновляют те же пулы. Пример полного цикла находится в binary
`qwc-engine/src/bin/scheduled.rs`. Preemption, swap и prefix snapshots в
текущий runtime намеренно не входят.
