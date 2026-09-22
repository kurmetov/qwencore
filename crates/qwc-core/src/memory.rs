//! Планирование VRAM под одну RTX 5090 (32 GB).
//!
//! Ключевое отличие от обычного трансформера: память под последовательность
//! состоит из ДВУХ независимых частей.
//!
//!   1. Рекуррентное состояние DeltaNet (48 слоёв) — константа на слот,
//!      не зависит от длины контекста: ~39 MB @ int8, ~81 MB @ bf16,
//!      ~157 MB @ fp32. Сверх слотов лежит одна расквантованная копия на
//!      движок, ~76 MB, по которой идёт префилл.
//!   2. Paged KV-кэш (16 слоёв) — линейно растёт: 32 KB на токен @ fp8.
//!
//! Точка равенства — около 2.5K токенов. Ниже неё concurrency упирается
//! в число слотов состояний, выше — в KV-кэш. Планировщик обязан учитывать
//! обе части, иначе admission control ошибается в разы.

use crate::arch::*;
use crate::dtype::Dtype;

pub const GB: u64 = 1_000_000_000;
pub const MIB: u64 = 1024 * 1024;

// ---------------------------------------------------------------------------
// Веса
// ---------------------------------------------------------------------------

/// Раскладка весов по форматам. Значения по умолчанию соответствуют
/// нашей загрузке чекпоинта QUASAR в text-only режиме.
#[derive(Debug, Clone, Copy)]
pub struct WeightPlan {
    /// Все linear-слои тела модели.
    pub body: Dtype,
    /// `lm_head` в чекпоинте не квантован (BF16); мы переводим его в FP8.
    pub lm_head: Dtype,
    /// `embed_tokens` — это lookup, точность некритична.
    pub embed: Dtype,
    /// MTP draft-голова для спекулятивного декодинга.
    pub mtp: Option<Dtype>,
    /// Vision tower. В text-only движке всегда `false`.
    pub vision: bool,
}

impl Default for WeightPlan {
    fn default() -> Self {
        Self {
            body: Dtype::Nvfp4,
            lm_head: Dtype::Fp8E4m3,
            embed: Dtype::Fp8E4m3,
            mtp: Some(Dtype::Bf16),
            vision: false,
        }
    }
}

impl WeightPlan {
    /// Как чекпоинт лежит на диске: lm_head и эмбеддинги в BF16,
    /// vision tower и MTP загружены. Это то, что грузит vLLM.
    pub fn as_shipped() -> Self {
        Self {
            body: Dtype::Nvfp4,
            lm_head: Dtype::Bf16,
            embed: Dtype::Bf16,
            mtp: Some(Dtype::Bf16),
            vision: true,
        }
    }

    /// Полный resident footprint весов. Эта величина нужна для VRAM-бюджета,
    /// но не является трафиком одного decode-шага: обычный шаг не читает MTP,
    /// vision tower и всю таблицу embedding.
    pub fn bytes(&self) -> u64 {
        let mut total = self.body.bytes(QUANTIZED_PARAMS);
        total += self.lm_head.bytes(LM_HEAD_PARAMS);
        total += self.embed.bytes(EMBED_PARAMS);
        if let Some(d) = self.mtp {
            total += d.bytes(MTP_PARAMS_EST);
        }
        if self.vision {
            total += Dtype::Bf16.bytes(VISION_PARAMS_EST);
        }
        total
    }

    /// Байты весов, которые обычный decode-шаг действительно читает из
    /// памяти. Тело модели и `lm_head` сканируются целиком, а embedding делает
    /// lookup одной строки на последовательность. MTP и vision в обычном
    /// decode не исполняются.
    pub fn decode_weight_bytes(&self, batch: usize) -> u64 {
        self.body.bytes(QUANTIZED_PARAMS)
            + self.lm_head.bytes(LM_HEAD_PARAMS)
            + self.embed.bytes(HIDDEN_SIZE * batch)
    }
}

// ---------------------------------------------------------------------------
// Кэш
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Формат KV-кэша для 16 full-attention слоёв.
    pub kv_dtype: Dtype,
    /// Формат рекуррентного состояния покоящихся слотов. Конфиг модели
    /// требует fp32; движок хранит int8 с масштабом на строку, а префилл
    /// идёт по отдельной bf16-копии (см. `state_work_bytes`).
    pub state_dtype: Dtype,
    /// Размер блока paged KV-кэша в токенах.
    pub block_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            kv_dtype: Dtype::Fp8E4m3,
            state_dtype: Dtype::Int8Row128,
            block_size: 64,
        }
    }
}

impl CacheConfig {
    /// Байт KV-кэша на один токен по всем full-attention слоям.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_dtype.bytes(KV_ELEMS_PER_TOKEN)
    }

    pub fn kv_bytes_per_block(&self) -> u64 {
        self.kv_bytes_per_token() * self.block_size as u64
    }

    /// Байт на слот состояния: рекуррентное состояние DeltaNet + conv-окно.
    /// Conv-состояние всегда fp32 — оно копеечное, экономить смысла нет.
    pub fn state_bytes_per_slot(&self) -> u64 {
        self.state_dtype.bytes(STATE_ELEMS_PER_SEQ) + Dtype::Fp32.bytes(CONV_ELEMS_PER_SEQ)
    }

    /// Расквантованная копия состояния одной последовательности. Она одна на
    /// движок, а не на слот, поэтому в стоимость слота не входит — но в
    /// бюджет карты входит, и на двух слотах она съедает всю экономию.
    pub fn state_work_bytes(&self) -> u64 {
        Dtype::Bf16.bytes(STATE_ELEMS_PER_SEQ)
    }

    /// Полная стоимость последовательности заданной длины.
    pub fn bytes_per_seq(&self, context_len: usize) -> u64 {
        let blocks = context_len.div_ceil(self.block_size) as u64;
        self.state_bytes_per_slot() + blocks * self.kv_bytes_per_block()
    }

    /// Длина контекста, при которой KV-кэш сравнивается по объёму
    /// с рекуррентным состоянием. Ниже неё «дорогая» часть — состояния.
    pub fn crossover_tokens(&self) -> usize {
        (self.state_bytes_per_slot() / self.kv_bytes_per_token()) as usize
    }
}

// ---------------------------------------------------------------------------
// Бюджет
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub total_vram: u64,
    /// Занято до старта движка: CUDA-контекст, драйвер, чужие процессы
    /// (на десктопе gnome-shell легко съедает больше гигабайта).
    pub reserved: u64,
    pub weights: u64,
    /// Активации, workspace CUTLASS, буферы сэмплинга, CUDA-графы.
    pub workspace: u64,
}

impl Budget {
    /// Память, доступная под KV-кэш и слоты состояний.
    pub fn cache_available(&self) -> u64 {
        self.total_vram
            .saturating_sub(self.reserved)
            .saturating_sub(self.weights)
            .saturating_sub(self.workspace)
    }

    /// Сколько последовательностей заданной длины помещается.
    pub fn max_concurrency(&self, cfg: &CacheConfig, context_len: usize) -> usize {
        let per_seq = cfg.bytes_per_seq(context_len);
        if per_seq == 0 {
            return 0;
        }
        (self.cache_available() / per_seq) as usize
    }

    /// Максимальный контекст при заданном числе последовательностей.
    pub fn max_context(&self, cfg: &CacheConfig, concurrency: usize) -> usize {
        if concurrency == 0 {
            return 0;
        }
        let per_seq = self.cache_available() / concurrency as u64;
        let for_kv = per_seq.saturating_sub(cfg.state_bytes_per_slot());
        (for_kv / cfg.kv_bytes_per_token()) as usize
    }

    pub fn fits(&self, cfg: &CacheConfig, concurrency: usize, context_len: usize) -> bool {
        cfg.bytes_per_seq(context_len) * concurrency as u64 <= self.cache_available()
    }
}

/// Бюджет RTX 5090 при работающем десктопе.
/// nvidia-smi показывает 32607 MiB всего и ~1238 MiB занято gnome-shell.
pub fn rtx5090(weights: u64, workspace: u64) -> Budget {
    Budget {
        total_vram: 32_607 * MIB,
        reserved: 2_100 * MIB, // desktop + CUDA-контекст
        weights,
        workspace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_budget() -> Budget {
        rtx5090(WeightPlan::default().bytes(), 2 * GB)
    }

    #[test]
    fn weight_plan_matches_checkpoint_size() {
        // На диске чекпоинт QUASAR занимает 20.6 GB.
        let shipped = WeightPlan::as_shipped().bytes() as f64 / 1e9;
        assert!((20.0..21.2).contains(&shipped), "as_shipped: {shipped} GB");

        // После наших преобразований — заметно меньше.
        let ours = WeightPlan::default().bytes() as f64 / 1e9;
        assert!((16.5..17.5).contains(&ours), "default: {ours} GB");

        let saved = shipped - ours;
        assert!(saved > 3.0, "экономия всего {saved} GB, ожидали >3");

        // Resident footprint включает MTP и полную embedding table, но
        // обычный batch-1 decode их не стримит целиком.
        let decode = WeightPlan::default().decode_weight_bytes(1) as f64 / 1e9;
        assert!(
            (14.5..15.5).contains(&decode),
            "decode traffic: {decode} GB"
        );
        assert!(
            decode < ours - 1.5,
            "resident {ours} GB vs decode {decode} GB"
        );
    }

    #[test]
    fn state_dominates_below_crossover() {
        let cfg = CacheConfig::default();
        let x = cfg.crossover_tokens();
        // 8-битное состояние равно KV-кэшу примерно на 1.3K токенов: вдвое
        // дешевле слот — вдвое ближе точка равенства, и диапазон, где
        // concurrency упирается в слоты, а не в KV, соответственно сузился.
        assert!((1000..1700).contains(&x), "точка пересечения: {x}");

        let wide = CacheConfig {
            state_dtype: Dtype::Bf16,
            ..cfg
        };
        assert!(
            wide.crossover_tokens() > x * 3 / 2,
            "bf16 {} против int8 {x}",
            wide.crossover_tokens()
        );

        // Ниже неё состояние дороже KV.
        let short: usize = 512;
        let blocks = short.div_ceil(cfg.block_size) as u64;
        assert!(cfg.state_bytes_per_slot() > blocks * cfg.kv_bytes_per_block());
    }

    #[test]
    fn fp32_state_shifts_crossover_much_higher() {
        let cfg = CacheConfig {
            state_dtype: Dtype::Fp32,
            ..CacheConfig::default()
        };
        let x = cfg.crossover_tokens();
        assert!((4200..5600).contains(&x), "fp32 точка пересечения: {x}");
    }

    #[test]
    fn benchmark_matrix_feasibility() {
        let b = default_budget();
        let cfg = CacheConfig::default();

        // Целевая матрица: всё до 32x2K должно помещаться.
        assert!(b.fits(&cfg, 1, 512));
        assert!(b.fits(&cfg, 1, 16 * 1024));
        assert!(b.fits(&cfg, 4, 2048));
        assert!(b.fits(&cfg, 8, 2048));
        assert!(b.fits(&cfg, 16, 2048));
        assert!(b.fits(&cfg, 32, 2048));
    }

    #[test]
    fn long_context_concurrency_limits() {
        let b = default_budget();

        let fp8 = CacheConfig::default();
        let n_fp8 = b.max_concurrency(&fp8, 32 * 1024);
        assert!((8..=14).contains(&n_fp8), "fp8 KV @32K: {n_fp8} seq");

        // NVFP4 KV-кэш почти удваивает concurrency на длинном контексте.
        let fp4 = CacheConfig {
            kv_dtype: Dtype::Nvfp4,
            ..CacheConfig::default()
        };
        let n_fp4 = b.max_concurrency(&fp4, 32 * 1024);
        assert!(
            n_fp4 as f64 / n_fp8 as f64 > 1.5,
            "fp4 {n_fp4} vs fp8 {n_fp8}"
        );
    }

    #[test]
    fn our_weight_savings_buy_concurrency() {
        let cfg = CacheConfig::default();
        let ours = rtx5090(WeightPlan::default().bytes(), 2 * GB);
        let vllm_like = rtx5090(WeightPlan::as_shipped().bytes(), 2 * GB);

        let ctx = 32 * 1024;
        let gain = ours.max_concurrency(&cfg, ctx) - vllm_like.max_concurrency(&cfg, ctx);
        assert!(gain >= 3, "выигрыш всего {gain} последовательностей");
    }
}
