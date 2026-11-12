# PROJECT_MEMORY.md — StrataSLM Persistent Agent Memory

> **READ THIS FIRST.** This file is the single source of truth for project continuity, architectural rules, chronological history, and lessons learned. Every AI agent or contributor working in this repository **must read this file before planning or making changes**.

---

## 1. Project Summary (From `PLAN.md`)

- **Project Name:** **StrataSLM** *(Alternative titles noted in `PLAN.md`: CogStrata, IronLattice)*
- **Mission:** Build a **Local AI Runtime & Reliable Agent Platform** where **system engineering compensates for Small Language Model (SLM) limits** (*"Make the system smarter, not the model"*).
- **Current Phase:** **Phases 0–5 Implementation, Evaluation Suite, Live Benchmarks, Profiling & Docs Verified** (see Sections 3–6 below for current status & telemetry).
- **Core Stack:** Python `3.12` (`uv`-managed single-process modular monolith), FastAPI + Uvicorn (single worker), SQLite (`WAL` + `FTS5` BM25 + `sqlite-vec`), `httpx`, Pydantic v2, `psutil`, quantized ONNX Runtime (`all-MiniLM-L6-v2`), and **Ollama** (`qwen3:1.7b` primary).
- **Primary Ollama Model Strategy:**
  - **Default:** `qwen3:1.7b` (`~1.4 GB` Q4_K_M, `~1.8–2.1 GB` RSS with `num_ctx=4096` & `OLLAMA_KV_CACHE_TYPE=q8_0`), using task-adaptive reasoning (`think: false` for fast extraction/routing/repair; `think: true` only for complex planning/recovery).
  - **Optional Tiers:** `qwen3:0.6b` (`~523 MB` fast router/smoke-test), `qwen3:4b` (`~2.5 GB` headless high-accuracy opt-in), `llama3.2:3b` (`~2.0 GB` stress-test baseline).
  - **Zero-Swap Embeddings:** Online retrieval uses SQLite `FTS5` + in-process CPU ONNX `all-MiniLM-L6-v2` (`~90–120 MB` RSS) so Ollama never unloads/reloads the chat model (`OLLAMA_MAX_LOADED_MODELS=1`).

### The 5 Project Pillars
1. **Inference Speed & CPU Optimization (`strata.inference`):** Prefix-canonical prompt layout (KV cache reuse), dynamic top-$k$ tool-schema pruning, 2-tier exact + semantic caching, CPU thread pinning, adaptive `think` routing, TTFT/`tok/s`/RSS benchmarks, and profiling reports.
2. **Reliable AI Agents (`strata.agents`):** Deterministic state machine (`PLAN → VALIDATE → AUTHORIZE → EXECUTE → VERIFY → COMMIT`), Pydantic JSON sanitization + max-2 targeted repair loop, pre-action policy guardrails (*AgentSpec*-style), untrusted data taint tracking (*CaMeL*-style), capability-capped subprocess sandbox, SQLite durable step journal (`resume` + idempotency + Saga rollbacks), and **Procedural Skill Distillation**.
3. **AI Infrastructure (`strata.server`, `strata.sdk`, `strata.plugins`):** OpenAI-compatible `/v1/chat/completions` + native `/v1/runs` SSE APIs, Python SDK (`StrataClient`, `@tool`), manifest-driven plugins + MCP adapter, CLI (`strata`), docs, and 3 lightweight sample apps.
4. **Cognitive Long-Term Memory (`strata.memory`):** CoALA 4-tier memory (Working, Episodic, Bi-Temporal Semantic Graph with contradiction invalidation, Procedural) in one `strata.db`, hybrid RRF (`FTS5 + sqlite-vec`), preemptible **Idle-Aware Sleep-Time Consolidation**, and evaluation vs baselines on curated LoCoMo & LongMemEval subsets.
5. **Continuous Performance Engineering (`strata.governor`, `strata.perf`):** `< 300 ms` lazy-import CLI startup, `psutil` 3-stage **Resource Governor** (OOM/lag guard), request coalescing/priority queue for `OLLAMA_NUM_PARALLEL=1`, and automated regression benchmarks.

---

## 2. Mandatory Rules for All Future Agents

> [!IMPORTANT]
> Every agent working on this workspace **MUST** obey all 5 rules below without exception.

1. **Explore `PROJECT_MEMORY.md` First:**
   - Before proposing edits, writing code, or running commands, read `PROJECT_MEMORY.md` in full (and `PLAN.md` for detailed specs) to understand what exists, current phase, prior decisions, and known pitfalls.
2. **Log Every Change with Date, Time & Reason:**
   - Every modification to the repository (files added, modified, deleted; dependencies installed; config/Modelfile changes; benchmark runs) **must** be appended to **Section 4 (Chronological Change Log)** with exact timestamp (`YYYY-MM-DD HH:MM +05:30` or local ISO), affected files, summary of change, and **why** it was done.
3. **Log Every Mistake & Lesson Learned:**
   - Whenever an error, OOM/slowdown, failed command, broken assumption, or regression occurs, record it immediately in **Section 5 (Mistakes & Lessons Learned)** with what went wrong, root cause, and the rule/fix adopted so future agents do not repeat it.
4. **Maintain Complete From-Day-One Continuity:**
   - Keep **Section 3 (Current Project State)** and the logs accurate after every task so any new agent joining mid-project understands the entire evolution from Day 1 (`PLAN.md`) to the present state.
5. **Strict 8GB RAM / CPU-Only Intel i5 Hardware Mandate:**
   - This project runs on an **8GB RAM, CPU-only laptop with an Intel Core i5 processor** (Windows AMD64, 12 logical CPUs).
   - **NEVER** add heavy dependencies (no Docker, Redis, Postgres, Neo4j, Qdrant server, PyTorch, TensorFlow, heavy browser/Node dev servers, or multi-GB models).
   - **NEVER** spawn unbounded parallel workers, large batch loads, or simultaneous Ollama models (`OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`).
   - Keep total AI stack RAM **$\le 3.0\text{ GB}$** (target **`2.0–2.5 GB`**), test suites fast, imports lazy, and execution lightweight so the laptop **never suffers OOM errors, swapping, thermal stalls, or UI lag**.

---

## 3. Current Project State Snapshot

- **Active Milestone:** Phases 0–5 Complete & Verified (Runtime, Gateway, SDK, Plugins, Cognitive Memory, Evals, Profiling & Docs)
- **Files in Workspace:**
  - `PLAN.md`, `PROJECT_MEMORY.md`, `README.md`, `pyproject.toml`, `.gitignore`, `uv.lock`
  - **Docs (`docs/`):** `ARCHITECTURE.md`, `PERFORMANCE_REPORT.md`
  - **Runtime (`src/strata/`):**
    - `config.py` — `StrataConfig` + `OLLAMA_HOST` normalization + 8 GB governor/RSS thresholds.
    - `cli.py` — Lazy-import CLI (`strata doctor`, `strata run`, `strata serve`, `strata profile [--save]`, `strata bench {threads,cold-start,eval,inference}`), median cold-start **`145.37 ms`** (`< 300 ms` target).
    - `governor/hardware.py`, `governor/resource_governor.py` — Win32 `GetLogicalProcessorInformationEx` hybrid P/E-core probe + cached-PID `probe_memory()` + sustained-window `psutil` CPU sampling + 3-stage `GREEN/YELLOW/RED` governor.
    - `storage/db.py` — Unified SQLite `3.50.4` WAL store (`strata.db`) with `sqlite-vec` (`v0.1.9`) extension loading and v1–v4 schemas (`l1_cache`, `l2_cache`, `agent_runs`, `agent_steps`, `memory_episodes` + FTS5, `memory_facts` + FTS5, `memory_links`, `memory_skills`), bounded 8 MiB page cache (`cache_size=-8000`).
    - `inference/client.py`, `tool_pruner.py`, `prompt.py`, `cache.py`, `router.py`, `engine.py` — Async Ollama streaming client with `_PriorityGate` (`RequestPriority`: `HIGH` > `NORMAL` > `BACKGROUND`), concurrent `_inflight` request coalescing, BM25 top-$k$ alphabetical tool pruner (`-69.9%` chars), `CanonicalPromptBuilder` (`74.4%` prefix reuse), L1 exact normalized cache (`L1_MEM` + `L1_SQLITE`), `L2SemanticCache` (`sqlite-vec` cosine + Python fallback), adaptive `think` router, and coordinated `InferenceEngine`.
    - `agents/schema.py`, `policy.py`, `journal.py`, `sandbox.py`, `runtime.py` — Zero-LLM JSON sanitizer + max-2 targeted Pydantic repair, `AgentSpec` pre-action policy + `CaMeL` untrusted taint tracking, Win32 Job Object (`128 MB` cap + `KILL_ON_JOB_CLOSE`) subprocess sandbox, durable step journal with idempotency + Saga rollback, recoverable `READ`/`COMPUTE` tool error observation feedback (`status="recovered_error"`), automatic procedural skill distillation (`save_skill` / `match_skill`), and `DeterministicAgentRuntime` (`PLAN → VALIDATE → AUTHORIZE → EXECUTE → VERIFY → COMMIT`).
    - `memory/embedder.py`, `store.py`, `consolidator.py` — Pluggable embedder (`HashingNgramEmbedder` 128-dim default + optional dynamic-padded quantized `OnnxMiniLMEmbedder` 384-dim via `onnxruntime`/`tokenizers`), CoALA 4-tier memory (Working, Episodic, Bi-Temporal Semantic Graph with contradiction invalidation & `as_of` recall, Procedural skills), Hybrid RRF (`FTS5 porter unicode61` + vector cosine + 1-hop A-MEM graph expansion + abstention), and governor-preemptible `SleepTimeConsolidator`.
    - `plugins/loader.py`, `sdk/client.py`, `server/app.py` — Manifest/subprocess plugin & MCP adapter (`PluginRegistry`, `load_manifest_file`, `adapt_mcp_tool`), Python SDK (`StrataClient.local`/`remote`, `@tool`), and FastAPI gateway (`create_app`: `/health`, `/v1/models`, `/v1/chat/completions` JSON & SSE, `/v1/runs`, `/v1/memory/*`).
    - `perf/bench.py`, `perf/evals.py` — `run_doctor`, `run_thread_sweep`, `measure_cold_start`, `eval_agent_reliability` (vs Naïve ReAct), `eval_memory_suite` (vs No-Memory/Sliding-Window/Vector-Only baselines), `eval_inference_optimizations`, `eval_concurrency_stress`, `eval_server_profile`, `profile_hotspots` (`cProfile`), and `run_live_inference_benchmark`.
  - **Examples (`examples/`):** `workspace_analyst.py`, `ops_helper.py`, `memory_companion.py`.
  - **Tests & Benchmarks:**
    - `tests/` — 12 test modules, **39 / 39 passing** (`13.93s` full `pytest -v`; `23 / 23` non-async pass in `1.67s` inside sandbox).
    - `benchmarks/results/thread_sweep_qwen3_1.7b_20261112_110223.json` — Live Phase 0 sweep on `qwen3:1.7b`.
    - `benchmarks/results/eval_suite_latest.json` — All 5 suites PASS (`Agent`: `100.0%` schema, `100.0%` policy block, `pass^3=3/3` vs naive `0/3`, skill distilled; `Memory`: `Recall@5=1.00` > `vec=0.88` > `win=0.00` = `none=0.00`, `Freshness=2/2`, `Abstention=3/3`, `0.21ms` mean, `51.5 MB` RSS; `Inference`: `-69.9%` tool prune, `74.4%` prefix reuse; `Concurrency`: `7/7` coalesced, `HIGH > BG`, `RED` shed OK; `Profile`: `186.5 ms` API boot, `0.01 ms` L1, `0.14 ms` L2, `59.8 MB` RSS).
    - `benchmarks/results/profile_latest.json` — Cold-start `175.1 ms` (profile run) / `241.7 ms` (5-run CLI sweep), API boot `146.5 ms`, `cProfile` (`80` iters = `40.8 ms`).
    - `benchmarks/results/live_inference_latest.json` — Live `qwen3:1.7b`: **`3.36x` TTFT speedup** (`1013.12 -> 301.81 ms`), **`0.104 ms` L1 (`28,972.9x`)**, **`1.162 ms` L2 (`2,593.1x`, sim=`0.906`)**, **`1920.04 MB` peak stack RSS**.
    - `benchmarks/results/embedder_ab_latest.json` — ADR-002 A/B benchmark (`Xenova/all-MiniLM-L6-v2` quantized `22.59 MB` in `.strata/models/all-MiniLM-L6-v2`): `Hashing(128)` margin `0.1046` / `L2_hit=0.00` / `Mem@1=0.67` vs `ONNX MiniLM(384)` margin **`0.4784`** (`+0.3738`) / `L2_hit=0.50` / `Mem@1=1.00` / `2.191 ms/txt` / RSS **`111.25 MB`** (`+61.25 MB`, `<= 180 MB`).
    - All 3 examples verified post-runtime-edits: `workspace_analyst.py` (`completed` in 3 steps with live `qwen3:1.7b` self-correction), `ops_helper.py` (`passed=true`, `Governor=GREEN`, `stack_rss_mb=1683.78`), `memory_companion.py` (`episodes_processed=3, facts_added=3, facts_updated=1, links_created=1`).
- **Detected Local Environment & Live Hardware Baseline (Verified 2026-11-12):**
  - CPU: Intel Core i5 (`AMD64`), **8 physical / 12 logical** (**4 P-cores** + **4 E-cores**); `psutil`/Win32 probe confirmed `4P + 4E`.
  - RAM on Dev Host: `16108 MiB` (`~15.73 GiB`) total, `~8.0 GiB` free; **8 GB target mandate (`<= 3072 MiB` stack ceiling, `<= 180 MiB` runtime ceiling) strictly enforced.**
  - Python `3.12.12` (`.venv` via `uv 0.10.2`), SQLite `3.50.4` (`FTS5` + `JSON1`), Ollama `0.40.16` (`qwen3:1.7b` `1.27 GiB` Q4_K_M, `qwen3:0.6b` `0.49 GiB` Q4_K_M pulled).
  - Live `qwen3:1.7b` (`think=False`, `num_ctx=4096`) Thread Sweep:
    - `2` threads: `17.98 tok/s` (`1933.2 MiB` stack RSS)
    - `4` threads (P-cores): `25.89 tok/s`, warm TTFT `69.5 ms` (`1934.1 MiB` stack RSS)
    - `6` threads: `26.88 tok/s`, warm TTFT `85.4 ms` (`1934.4 MiB` stack RSS)
    - `8` threads (all 8 physical cores): **`28.82 tok/s`**, warm TTFT `64.8 ms` (`1935.8 MiB` stack RSS)
    - `12` threads (with SMT): `28.82 tok/s`, warm TTFT `61.4 ms` (`1937.0 MiB` stack RSS) — zero throughput gain over `8` physical cores.
  - Live `strata run` (`qwen3:1.7b`, `think=False`): uncached `wall=4389.0 ms` (`ttft=2115.5 ms`, `25.5 t/s`) vs cached `L1_SQLITE` hit **`2.95 ms`** (`~1487x` speedup).

---

## 4. Chronological Change Log

| Timestamp | Phase | Files / Scope | Change Summary | Why |
|---|---|---|---|---|
| `2026-11-11 18:08 +05:30` | Planning | `PLAN.md` (created) | Researched 2024–2026 literature & open-source stacks; verified Ollama model specs; wrote complete architecture, 8GB RAM budget, 5-pillar design, and Phases 0–5 plan for **StrataSLM**. | Establish a rigorous, OOM-safe engineering blueprint before writing any code. |
| `2026-11-11 18:11 +05:30` | Planning | `PROJECT_MEMORY.md` (created) | Created persistent agent memory file containing project summary, mandatory agent rules, hardware mandate, change log, and lessons template. | Ensure all future agents have full Day-1 context, log all changes/learnings, and strictly protect the 8GB CPU-only laptop. |
| `2026-11-12 10:50 +05:30` | Phase 0 | `pyproject.toml`, `.gitignore`, `README.md`, `uv.lock`, `src/strata/{config.py,cli.py,governor/*,inference/client.py,perf/bench.py}`, `tests/test_{config,governor,inference_client,perf_cli}.py` | Initialized minimal `uv` Python 3.12 project (`httpx`, `psutil`, `pydantic` — 20 packages total), implemented Win32 P/E-core probe + 3-stage `ResourceGovernor`, async `OllamaClient`, `strata doctor`, `strata bench threads`, and `strata bench cold-start`. | Establish Phase 0 baseline runtime with `< 300 ms` lazy-import startup and strict 8 GB telemetry. |
| `2026-11-12 10:55 +05:30` | Phase 0 | Ollama models `qwen3:0.6b` & `qwen3:1.7b` | Pulled `qwen3:0.6b` (`0.49 GiB`) and `qwen3:1.7b` (`1.27 GiB`) via `ollama pull`; verified via `strata doctor` (`Ollama v0.40.16`). | Provide local primary (`1.7b`) and fast-router (`0.6b`) models per ADR-001. |
| `2026-11-12 10:57 +05:30` | Phase 1 | `src/strata/storage/db.py`, `src/strata/inference/{tool_pruner,prompt,cache,router,engine}.py`, `src/strata/cli.py`, `tests/test_{phase1_cache_prompt,inference_engine}.py` | Implemented SQLite WAL store, BM25 top-k tool pruner with alphabetical KV-canonical sorting, `CanonicalPromptBuilder`, `L1ExactCache` (`L1_MEM` + `L1_SQLITE`), `AdaptiveRouter`, `InferenceEngine`, and `strata run`. | Deliver KV prefix reuse, token-budgeted prompts, sub-ms L1 cache hits, and adaptive `think` routing. |
| `2026-11-12 11:00 +05:30` | Phase 2 | `src/strata/storage/db.py` (v2), `src/strata/agents/{schema,policy,journal,sandbox,runtime}.py`, `tests/test_agents_{schema_policy,runtime_sandbox}.py` | Implemented zero-LLM JSON sanitizer + max-2 repair loop, `AgentSpec` pre-action policy + `CaMeL` untrusted taint guard, Win32 Job Object (`128 MB` cap) subprocess sandbox, SQLite durable step journal (`agent_runs`/`agent_steps`), and `DeterministicAgentRuntime`. | Guarantee deterministic, loop-guarded, injection-resistant, crash-resumable agent execution on 1.7B SLMs. |
| `2026-11-12 11:04 +05:30` | Phase 0 & 3 | `src/strata/storage/db.py` (v3), `src/strata/memory/{store,consolidator}.py`, `tests/test_memory_engine.py`, `benchmarks/results/thread_sweep_qwen3_1.7b_20261112_110223.json` | Ran live Phase 0 thread sweep on `qwen3:1.7b` (`2,4,6,8,12` threads); implemented CoALA 4-tier `CognitiveMemoryStore` (`porter unicode61` FTS5 + vector cosine + 1-hop A-MEM graph RRF, bi-temporal invalidation, abstention) and governor-preemptible `SleepTimeConsolidator`; verified all 26 tests + live `strata run` (`2.95 ms` L1 hit). | Lock in real CPU thread/RSS baselines and establish the bi-temporal cognitive memory engine before Phase 1b/3b/4 integration. |
| `2026-11-12 11:22 +05:30` | Phase 1b, 3b, 4 & 5 | `pyproject.toml`, `uv.lock`, `src/strata/{storage/db.py,memory/embedder.py,memory/store.py,inference/{cache,engine}.py,agents/runtime.py,plugins/*,sdk/*,server/*,perf/evals.py,cli.py}`, `examples/*`, `tests/test_{l2_coalescing_embedder,server_sdk_plugins,eval_suite,agents_runtime_sandbox}.py`, `benchmarks/results/eval_suite_latest.json`, `README.md` | Added `fastapi`, `uvicorn`, `sqlite-vec`, and optional `[onnx]` (`onnxruntime`, `tokenizers`); built L2 semantic cache (`sqlite-vec` + fallback), optional quantized ONNX MiniLM embedder, single-flight request coalescing, recoverable `READ`/`COMPUTE` agent tool error self-correction, FastAPI gateway (`/v1/chat/completions`, `/v1/runs`, `/v1/memory/*`), `StrataClient` + `@tool`, manifest/MCP plugin loader, `strata {serve,profile,bench eval}`, and 3 examples; verified `34/34` tests (`20/20` non-async in sandbox), `strata bench eval --save`, `strata profile`, `examples/ops_helper.py`, and live `examples/workspace_analyst.py` on `qwen3:1.7b`. | Complete semantic caching, API/SDK/plugin surface, and automated reliability/memory/RSS/boot regression gates within the 8 GB envelope. |
| `2026-11-12 11:44 +05:30` | Phase 5 & Closure | `src/strata/{inference/{client,engine}.py,agents/runtime.py,memory/embedder.py,governor/{hardware,resource_governor}.py,perf/{bench,evals}.py,server/app.py,sdk/client.py,cli.py}`, `docs/{ARCHITECTURE,PERFORMANCE_REPORT}.md`, `README.md`, `tests/test_{eval_suite,inference_client,agents_runtime_sandbox,perf_cli}.py`, `benchmarks/results/{eval_suite,profile,live_inference}_latest.json` | Added `_PriorityGate` (`HIGH > NORMAL > BACKGROUND`) single-flight scheduler; wired procedural skill distillation/matching in `DeterministicAgentRuntime`; added PID-cached `probe_memory()` + `>=0.5s` sustained CPU window (cutting full pytest time `76.76s -> 10.68s`); switched `OnnxMiniLMEmbedder` to dynamic batch padding; expanded `evals.py` with Naïve ReAct & No-Memory/Sliding-Window/Vector-Only baselines, prefix/pruning checks, concurrency/OOM stress, `cProfile` hotspots, and `strata bench inference`; generated `ARCHITECTURE.md` and `PERFORMANCE_REPORT.md`; verified `37/37` tests, all CLI benchmarks, and all 3 examples live. | Complete all remaining Phase 1–5 deliverables and Section 8 benchmark gates with real profiling fixes and zero external bloat. |
| `2026-11-12 12:17 +05:30` | ADR-002 & GitHub CI | `src/strata/{memory/embedder.py,perf/evals.py,cli.py}`, `tests/test_{l2_coalescing_embedder,perf_cli,server_sdk_plugins}.py`, `pyproject.toml`, `.github/workflows/ci.yml`, `PLAN.md`, `README.md`, `docs/PERFORMANCE_REPORT.md`, `PROJECT_MEMORY.md` | Added `fetch_minilm_onnx()` + `strata models fetch-minilm` and `eval_embedder_ab()` + `strata bench embedders`; downloaded real `Xenova/all-MiniLM-L6-v2` quantized ONNX (`22.59 MB`) and proved **`4.6x` separation margin gain** (`0.4784` vs `0.1046`) and **`1.00` `Mem@1`** at **`111.25 MB` RSS** (`<= 180 MB`); added NUL-byte plugin sandbox rejection assertion, `[project.urls]`, GitHub Actions CI (`ubuntu-latest` + `windows-latest`), and `git init -b main`; verified `39/39` tests, `uv lock --check`, `eval`, `profile`, `cold-start` (`241.7 ms`), and `bench embedders`. | Validate ADR-002 in-process quantized ONNX embeddings live under 180 MB RSS and prepare StrataSLM for clean CI-tested GitHub publication. |
| `2026-11-12 12:27 +05:30` | Final Polish & CI Pass | `src/strata/{agents/runtime.py,cli.py}`, `tests/test_{agents_runtime_sandbox,perf_cli}.py`, `README.md`, `docs/ARCHITECTURE.md`, `PROJECT_MEMORY.md` | Verified GitHub Actions CI (`windows-latest` + `ubuntu-latest`) passed (`success` on run `43170839614`); re-tested all 3 `examples/*.py` live with ONNX MiniLM active (`ops_helper`, `memory_companion`, `workspace_analyst`), fixed live `qwen3:1.7b` duplicate-read halt in `workspace_analyst` via 1-step soft-nudge loop guard (`status=completed`, `steps=2`), added `strata --version`, cleaned README & ARCHITECTURE CLI docs, and re-verified `39/39` tests (`14.68s`), `23/23` sync tests (`1.72s`), `strata bench eval` (`all PASS`), and cold-start (`244.0 ms` unsandboxed / `263.7 ms` sandboxed). | Ensure live sample apps, CLI UX, documentation, and remote GitHub CI are 100% green before final handoff. |

*(Future agents: append new entries chronologically at the bottom of this table.)*

---

## 5. Mistakes, Pitfalls & Lessons Learned

### Observed Environment & Planning Lessons
- **`2026-11-11 18:03 +05:30` — Windows hardware inspection caveats (`wmic` / `Get-CimInstance` / `systeminfo`):**
  - *What happened:* `wmic` is absent on this Windows setup, and `Get-CimInstance` failed inside the restricted execution environment (`Access to a CIM resource was not available`), while `systeminfo` exited with an error.
  - *Lesson:* For lightweight runtime/telemetry checks inside Python or PowerShell without CIM/WMIC failures, rely on Python standard library + `psutil` / Windows `GlobalMemoryStatusEx` & `GetLogicalProcessorInformationEx` via `ctypes` rather than spawning `wmic` or `systeminfo` subprocesses.
- **`2026-11-11 18:05 +05:30` — Avoiding multi-model Ollama RAM thrashing & oversized thread counts:**
  - *What happened (Architectural Analysis):* Loading a 1.7B–4B Ollama chat model alongside an Ollama embedding model exceeds comfortable 8GB RAM headroom or triggers 2–5 s disk reload swaps; also, using all 12 logical CPU threads on an Intel Core i5 mobile CPU degrades memory-bandwidth-bound inference.
  - *Lesson / Rule:* Keep `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NUM_PARALLEL=1`, run embeddings in-process via quantized ONNX (`all-MiniLM-L6-v2`) + SQLite `FTS5`, and tune `num_thread` to physical cores (`4` P-cores for background-friendly `25.89 tok/s`, or `8` physical cores for peak `28.82 tok/s`) rather than `12` SMT threads.
- **`2026-11-11 18:05 +05:30` — Choosing `qwen3:1.7b` over `qwen3.5:2b` for text-only 8GB CPU agents:**
  - *What happened (Model Audit):* `qwen3.5:2b` (~2.7 GB) bundles multimodal vision components, making its footprint nearly double `qwen3:1.7b` (~1.4 GB).
  - *Lesson / Rule:* Default to `qwen3:1.7b` for text/tool-calling agent workflows on 8GB RAM and exploit its `think: false/true` toggle for speed.
- **`2026-11-12 10:52 +05:30` — Python `asyncio` event loop creation hangs inside Windows Terminal Sandbox (`BypassSandbox: false`):**
  - *What happened:* Calling `asyncio.run(...)` or `pytest-asyncio` inside the network-isolated Windows sandbox hung indefinitely inside `socket._fallback_socketpair()` -> `accept()` in `asyncio/proactor_events.py:_make_self_pipe`.
  - *Root Cause:* Python 3.12's `ProactorEventLoop` / `SelectorEventLoop` creates a `127.0.0.1` loopback TCP `socketpair()` self-pipe on startup, which blocks in the restricted sandbox where loopback sockets are disallowed.
  - *Lesson / Rule:* Run pure synchronous tests inside the sandbox via `uv run --no-sync pytest -m "not asyncio"`, and execute async tests or live Ollama CLI calls (`strata doctor`, `strata bench threads`, `strata run`, full `pytest`) outside the sandbox (`BypassSandbox: true`).
- **`2026-11-12 11:02 +05:30` — Empirical `qwen3:1.7b` thread sweep & thread-selection tie-breaking:**
  - *What happened:* On `4P + 4E` (8 physical / 12 logical) CPU, `num_thread=8` and `num_thread=12` both produced `28.82 tok/s`, while naive `max(eval_tok_s, -warm_ttft)` picked `12` due to a `3.4 ms` noise difference.
  - *Lesson / Fix:* Updated `run_thread_sweep()` in `src/strata/perf/bench.py` to prefer the lowest `num_thread` within 1% of peak `eval_tok_s`, preventing unnecessary SMT/E-core saturation. Also noted that `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_KV_CACHE_TYPE`, and `OLLAMA_FLASH_ATTENTION` are currently unset in the host shell (`strata doctor` emits `[WARN]`).
- **`2026-11-12 11:03 +05:30` — Memory recall gap between `"live"` vs `"lives_in"` and entity-linked facts:**
  - *What happened:* `extract_rule_facts()` missed capitalized `"My name is ..."` and over-captured `"Chennai and I prefer..."`, and `recall_facts("Where does Maya live?")` missed `("user", "lives_in", "Hyderabad")` because `"live"` didn't match `"lives_in"` and `"Maya"` resided in a separate linked fact `("user", "name", "Maya")`.
  - *Lesson / Fix:* Added case-insensitive name extraction + conjunction boundary stop in `consolidator.py`, enabled SQLite FTS5 `tokenize='porter unicode61'` + stop-word filtering, normalized predicates in `fact_text`, and added 1-hop `memory_links` A-MEM neighbor expansion in `recall_facts()`.
- **`2026-11-12 11:15 +05:30` — `sqlite-vec` vector dimension mismatch & NULL guards:**
  - *What happened:* When mixing `HashingNgramEmbedder` (128-dim) and `OnnxMiniLMEmbedder` (384-dim) or encountering empty/NULL `embedding_blob` rows, `vec_distance_cosine(embedding_blob, ?)` can error or fail comparison.
  - *Lesson / Fix:* Guarded `vec_distance_cosine` queries in `store.py` and `cache.py` with `AND length(embedding_blob) = ?` and explicit `is not None` checks.
- **`2026-11-12 11:18 +05:30` — Live `qwen3:1.7b` tool path misguesses aborting agent runs:**
  - *What happened:* Running `examples/workspace_analyst.py` on live `qwen3:1.7b` failed on step 1 when the model passed `"subdir": "strata"` instead of `"src/strata"` (`FileNotFoundError`), immediately aborting the run.
  - *Lesson / Fix:* Updated `src/strata/agents/runtime.py` so recoverable `READ`/`COMPUTE` exceptions (`FileNotFoundError`, `NotADirectoryError`, `ValueError`, `TypeError`, `KeyError`, `< 2` consecutive errors) append a `ToolError` observation to `history` and record `status="recovered_error"` in the journal. Re-running live `examples/workspace_analyst.py` succeeded (`status="completed"`, `steps_executed=3`, self-corrected `"strata"` → `"src/strata"`).
- **`2026-11-12 11:40 +05:30` — Uncached `psutil.process_iter` & sub-quantum CPU spikes on Windows:**
  - *What happened:* Calling `psutil.process_iter` inside `probe_memory()` on every `ResourceGovernor.evaluate()` added ~200–400 ms per call (stretching pytest to `76.76s`), and back-to-back `psutil.cpu_percent(interval=None)` calls during bursts hit `100.0%` scheduling spikes, tripping spurious `RED` shedding / `L1_MEM` eviction in tests.
  - *Lesson / Fix:* Added a `5.0s` Ollama PID discovery TTL in `hardware.py` (`_OLLAMA_PID_CACHE`, probing known PIDs directly + always reading live `virtual_memory()` & self RSS) and a `>= 0.5s` sustained CPU sampling window in `resource_governor.py`. Full `pytest -v` dropped **7x** (`76.76s -> 10.68s`) and eliminated all transient CPU false alarms.

- **`2026-11-12 12:15 +05:30` — Dual-test benchmark differentiation (`Mem@5` ceiling vs `Mem@1`) & parallel cold-start noise:**
  - *What happened:* With only 12 total memory items, hybrid FTS+vec `Recall@5` saturated at `1.00` for both `HashingNgram` and `ONNX MiniLM`, and running `strata bench cold-start` concurrently with heavy ONNX benchmarks raised subprocess cold-start from `~241 ms` to `345 ms`.
  - *Lesson / Fix:* Added strict `memory_recall_at_1` alongside `memory_recall_at_5` + matched cross-vocabulary L2 paraphrases in `eval_embedder_ab()` (`ONNX MiniLM` lifts `Mem@1` `0.67 -> 1.00`, `L2_hit` `0.00 -> 0.50`, margin `0.1046 -> 0.4784` at `111.25 MB` RSS), and always measure `strata bench cold-start` solo (`241.7 ms` median inside sandbox, `175.1 ms` in unsandboxed `profile`).

- **`2026-11-12 12:24 +05:30` — Live `qwen3:1.7b` single-repeat tool calls hitting strict immediate loop-guard:**
  - *What happened:* When re-running `examples/workspace_analyst.py` live on `qwen3:1.7b`, the model called `read_snippet` twice in a row with identical arguments and got halted immediately (`status="halted"`).
  - *Lesson / Fix:* Added `"or repeat an identical tool call"` to `AGENT_SYSTEM_PROMPT` and added a 1-retry soft nudge in `DeterministicAgentRuntime.run()` (skips re-executing the duplicate tool, appends a `[tool] Already executed with identical arguments...` reminder, and only halts on the persistent repeat). Live `examples/workspace_analyst.py` immediately succeeded (`status="completed"`, `steps_executed=2`).

*(Future agents: whenever a command fails, a dependency bloats RAM, or an optimization assumption fails, log it here with timestamp, root cause, and fix.)*

---

## 6. Next Planned Milestone

- **Post-v1 Extensions (Optional Enhancements Beyond v0.1.0):**
  1. Integrate external full LoCoMo / LongMemEval JSON trace datasets and live MCP stdio server plugin demos.
  2. Add wheel build & tagged release publishing to `.github/workflows/ci.yml` once `v0.1.x` releases begin.
