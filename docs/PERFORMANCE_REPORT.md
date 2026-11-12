# StrataSLM Performance & 8 GB Safety Report

**Hardware Baseline:** Intel Core i5 (`AMD64`, 8 physical cores / 12 logical — **4 P-cores + 4 E-cores**), Python `3.12.12`, SQLite `3.50.4` (`WAL` + `FTS5` + `sqlite-vec v0.1.9`), Ollama `0.40.16` (`qwen3:1.7b` Q4_K_M, `qwen3:0.6b` Q4_K_M).

---

## 1. Gate Summary vs `PLAN.md` §8 Targets

| Suite | Metric | Target (`PLAN.md`) | Measured Result | Status |
|---|---|---|---:|---|
| **Startup & API** | CLI cold-start median | `< 300 ms` | **`145.37 ms`** | PASS |
| **Startup & API** | FastAPI `create_app()` boot | `< 800 ms` | **`141.47 ms`** (`eval`) / **`187.81 ms`** (`profile`) | PASS |
| **Inference & Cache** | Prefix-warm TTFT speedup (`qwen3:1.7b`) | `>= 2.0x` | **`3.36x`** (`1013.1 ms -> 301.8 ms`) | PASS |
| **Inference & Cache** | L1 Exact cache hit latency (`L1_MEM`) | `< 15 ms` | **`0.011 ms`** (`eval`) / **`0.104 ms`** (live) | PASS |
| **Inference & Cache** | L2 Semantic cache hit latency (`L2_SEMANTIC`) | `< 15 ms` | **`0.118 ms`** (`eval`) / **`1.162 ms`** (live) | PASS |
| **Inference & Cache** | Tool pruning prefill savings (10 → 3 tools) | `50–80%` | **`69.9%`** | PASS |
| **Inference & Cache** | Canonical KV prefix token reuse | `>= 70%` | **`74.4%`** (`prefix_hash_stable=True`) | PASS |
| **Agent Reliability** | Schema validity (malformed SLM JSON + repair) | `>= 95%` | **`100.0%`** (`5/5`) | PASS |
| **Agent Reliability** | Policy & taint injection block rate | `100%` | **`100.0%`** (`5/5`) | PASS |
| **Agent Reliability** | `pass^3` vs Naïve ReAct baseline | `> +20%` gain | **`3/3` vs `0/3` (`+100.0%`)** + skill distilled | PASS |
| **Cognitive Memory** | Recall@5 vs Baselines | Higher than baselines | **Strata `1.00` > Vector `0.88` > Sliding `0.00` = None `0.00`** | PASS |
| **Cognitive Memory** | Freshness (`as_of`) & Abstention | Full pass | **`2/2` Freshness**, **`3/3` Abstention** | PASS |
| **Cognitive Memory** | Mean / Max memory query latency | `< 30 ms` | **`0.22 ms`** mean / **`0.42 ms`** max | PASS |
| **Memory Footprint** | Strata Memory Runtime RSS | `<= 150 MB` | **`51.63 MB`** | PASS |
| **Memory Footprint** | Strata Runtime + FastAPI RSS | `<= 180 MB` | **`59.90 MB`** (`eval`) / **`36.70 MB`** (`ops_helper`) | PASS |
| **8 GB Stack Budget** | Total AI Stack RSS (Strata + Ollama `qwen3:1.7b`) | `<= 2560 MB` target / `<= 3072 MB` ceiling | **`1920.04 MB`** peak live / **`1683.78 MB`** idle (`ops_helper`) | PASS |
| **Concurrency & OOM** | Coalescing + Priority Lanes + RED Shed | Zero OOM, graceful shed | **`7/7` coalesced (`1` Ollama call)**, `HIGH > BG`, `RED` shed OK | PASS |

---

## 2. Live `qwen3:1.7b` CPU Thread Sweep (`num_ctx=4096`, `think=false`)

Measured via `strata bench threads --model qwen3:1.7b --threads 2,4,6,8,12` (local gitignored output: `benchmarks/results/thread_sweep_qwen3_1.7b_20261112_110223.json`):

| `num_thread` | Cold TTFT (ms) | Warm TTFT (ms) | Prompt (tok/s) | Eval (tok/s) | Ollama RSS (MiB) | Total Stack RSS (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| `2` | `2402.2` | `82.9` | `944.5` | `17.98` | `1891.4` | `1933.3` |
| `4` *(P-cores)* | `2223.0` | `69.5` | `1270.1` | `25.89` | `1892.3` | `1934.1` |
| `6` | `2206.4` | `85.4` | `1035.0` | `26.88` | `1892.5` | `1934.4` |
| `8` *(all physical)* | `2193.7` | `64.8` | `1361.3` | **`28.82`** | `1893.9` | `1935.8` |
| `12` *(SMT)* | `2149.3` | `61.4` | `1449.1` | `28.82` | `1895.0` | `1937.0` |

**Engineering Takeaway:** Memory bandwidth saturates at `8` physical cores (`28.82 tok/s`), while `4` P-cores deliver `25.89 tok/s` (`90%` of peak) leaving E-cores free for Windows UI responsiveness. Thread selection was updated to break ties within 1% of peak throughput toward lower `num_thread`.

---

## 3. Live `qwen3:1.7b` Caching & Prefix Reuse (`strata bench inference`)

Measured via `strata bench inference --model qwen3:1.7b --save` (local gitignored output: `benchmarks/results/live_inference_latest.json`):

- **Uncached Ollama (`max_tokens=48`, 200-tok system prefix):** `wall = 3013.18 ms`, `ttft = 1013.12 ms`, `24.04 tok/s`
- **Strata Turn 2 Shared Canonical Prefix:** `wall = 2398.98 ms`, `ttft = 301.81 ms` (**`3.36x` faster TTFT**)
- **Strata L1 Cache Hit (`L1_MEM`):** **`0.104 ms`** (**`28,972.9x`** faster than uncached wall latency)
- **Strata L2 Semantic Hit (`L2_SEMANTIC`, `sim = 0.906`):** **`1.162 ms`** (**`2,593.1x`** faster)
- **Peak Stack RSS:** **`1920.04 MB`** (`<= 3072 MB` hard ceiling)

---

## 4. CPU Hotspot Profile & Runtime Optimizations Applied

Measured via `strata profile --save` (`80` iterations of tool pruning, canonical prompt packing, FTS5 + vector hybrid memory recall, and JSON sanitization = **`41.88 ms`** total, **`~0.52 ms/iter`**; local gitignored output: `benchmarks/results/profile_latest.json`):

| Function | Location | Calls | Self Time (ms) | Cum Time (ms) |
|---|---|---:|---:|---:|
| `sqlite3.Connection.execute` | stdlib C | `240` | `7.27` | `7.27` |
| `ToolPruner.select` | `tool_pruner.py:123` | `80` | `4.36` | `15.45` |
| `json.encoder.iterencode` | `encoder.py:205` | `1120` | `3.16` | `3.16` |
| `HashingNgramEmbedder.encode_one` | `embedder.py:48` | `80` | `2.57` | `5.86` |
| `CognitiveMemoryStore.recall_facts` | `store.py:252` | `80` | `2.55` | `19.29` |

**Hot-Path Fixes Identified & Landed in Phase 5:**
1. **`probe_memory()` Process Scan Caching (`hardware.py`):** Full `psutil.process_iter` scans previously ran on every `ResourceGovernor.evaluate()`, adding ~200 ms per inference/consolidation step on Windows and slowing the pytest suite to `76.76s`. Caching discovered Ollama PIDs (`ollama_scan_ttl=5.0s`) while polling only cached PIDs + live `virtual_memory()` dropped full pytest runtime **7x** (`76.76s -> 10.68s`) and dropped L1/L2 engine overhead to **`< 0.2 ms`**.
2. **Sustained CPU Sampling Window (`resource_governor.py`):** Enforced a `>= 0.5s` sampling window on `psutil.cpu_percent(interval=None)` so sub-millisecond single-quantum scheduling spikes do not spuriously trip `YELLOW`/`RED` pressure shedding.
3. **Dynamic Batch Padding (`embedder.py`):** Switched `OnnxMiniLMEmbedder` from fixed `length=max_length` (256) padding to dynamic longest-in-batch padding for lower CPU latency and temporary tensor RAM when ONNX MiniLM weights are supplied.

---

## 5. Embedder A/B Benchmark (ADR-002: Hashing vs Quantized ONNX MiniLM)

Measured via `strata models fetch-minilm` (`22.59 MB` total for `Xenova/all-MiniLM-L6-v2` quantized ONNX + `tokenizer.json`) followed by `strata bench embedders --save --require-onnx` (`6` cross-vocabulary paraphrase pairs vs `6` unrelated distractor pairs; local gitignored output: `benchmarks/results/embedder_ab_latest.json`):

| Embedder | Dim | Paraphrase Cosine | Unrelated Cosine | Separation Margin | L2 Cache Hit Rate (fp=0) | Memory Recall@1 | Encode Latency | Process RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`HashingNgramEmbedder`** *(zero-weight fallback)* | `128` | `0.1917` | `0.0871` | `0.1046` | `0.00` | `0.67` | **`0.062 ms/txt`** | `50.00 MB` |
| **`OnnxMiniLMEmbedder`** *(`all-MiniLM-L6-v2` INT8)* | `384` | **`0.5379`** | **`0.0595`** | **`0.4784`** (`+0.3738`) | **`0.50`** (`+0.50`) | **`1.00`** (`+0.33`) | `2.191 ms/txt` | **`111.25 MB`** (`+61.25 MB`) |

**Decision Validated:** Quantized in-process ONNX MiniLM delivers **`4.6x` higher semantic separation margin** (`0.4784` vs `0.1046`), turns cross-vocabulary L2 semantic caching from `0% -> 50%` at zero false positives, and lifts top-1 cognitive memory recall from `67% -> 100%`, all while keeping process RSS at **`111.25 MB`** (`<= 180 MB` ceiling) and avoiding a separate Ollama embedding model swap.
