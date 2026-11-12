# StrataSLM

[![StrataSLM CI](https://github.com/SiddardhaShayini/StrataSLM/actions/workflows/ci.yml/badge.svg)](https://github.com/SiddardhaShayini/StrataSLM/actions/workflows/ci.yml)

> *System Intelligence Over Model Scale* — a featherweight, deterministic local AI runtime & cognitive memory engine engineered for an 8 GB RAM, CPU-only laptop.

See [PLAN.md](PLAN.md) for the master blueprint, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design, [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md) for measured 8 GB benchmarks, and [PROJECT_MEMORY.md](PROJECT_MEMORY.md) for continuity logs and hardware constraints.

## Quickstart

```powershell
# Sync lightweight Python 3.12 environment (add --extra onnx for quantized ONNX MiniLM)
uv sync

# Inspect hardware topology, Ollama readiness, and 8 GB RAM budget
uv run strata doctor

# Run CPU thread sweep benchmark on qwen3:1.7b
uv run strata bench threads --model qwen3:1.7b --threads 2,4,6,8,12 --save

# Run automated reliability (pass^3, injection), memory baselines, concurrency, and profile evals
uv run strata bench eval --save

# Benchmark live Ollama uncached vs prefix-warm vs L1 & L2 semantic cache hits
uv run strata bench inference --model qwen3:1.7b --save

# Profile CLI cold-start, API boot, cache latency, memory latency, cProfile hotspots, and RSS
uv run strata profile --save

# Run a prompt through InferenceEngine (prefix packer, router, L1+L2 cache)
uv run strata run "Give three crisp benefits of SQLite WAL mode for local AI."

# Start single-worker OpenAI-compatible FastAPI gateway (/v1/chat/completions, /v1/runs, /v1/memory/*)
uv run strata serve --bind 127.0.0.1 --port 8000 --sleep-bg
```

## Sample Apps

- `examples/workspace_analyst.py` — Workspace file analyst using `@tool` + `StrataClient.local`
- `examples/ops_helper.py` — 8 GB hardware & `ResourceGovernor` telemetry helper
- `examples/memory_companion.py` — CoALA bi-temporal recall & sleep-time consolidation

## Recommended Ollama Environment (8 GB Safety)

```powershell
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_FLASH_ATTENTION = "1"
```

## Optional ONNX MiniLM Embeddings (ADR-002)

Strata defaults cleanly to zero-footprint 128-dim character n-gram embeddings and automatically upgrades to quantized in-process 384-dim `all-MiniLM-L6-v2` (ONNX Runtime) when weights are present:

```powershell
uv sync --extra onnx
uv run strata models fetch-minilm
uv run strata bench embedders --require-onnx --save
```

Weights are stored locally in `.strata/models/all-MiniLM-L6-v2/` (gitignored).

## Verification & Benchmarks

```powershell
uv run pytest -v
uv run strata bench eval --save
uv run strata bench cold-start --runs 5
uv run strata profile --save
```

## Documentation

- [Architecture & ADRs](docs/ARCHITECTURE.md)
- [Performance & 8 GB Safety Report](docs/PERFORMANCE_REPORT.md)
- [Master Plan](PLAN.md)
- [Project Memory](PROJECT_MEMORY.md)
