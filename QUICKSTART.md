# StrataSLM Step-by-Step Quickstart Guide

This guide walks through every step to install, configure, benchmark, and build with **StrataSLM** on an 8 GB RAM, CPU-only machine (Windows 11 or Linux/macOS).

---

## Step 1: Check Prerequisites

1. **Python `3.12.x`** (`pyproject.toml` requires `>=3.12,<3.13`).
2. **`uv`** package manager (`https://docs.astral.sh/uv/`).
3. **Ollama** installed and running locally (`http://127.0.0.1:11434`).

Verify your tooling before cloning:

```powershell
python --version
uv --version
ollama --version
```

---

## Step 2: Clone & Sync Dependencies

Clone the repository and choose your dependency profile:

```powershell
git clone https://github.com/SiddardhaShayini/StrataSLM.git
cd StrataSLM
```

**Option A — Base Featherweight Runtime (128-dim fallback embeddings, zero ML bloat):**

```powershell
uv sync
```

**Option B — Base Runtime + Quantized ONNX MiniLM Embeddings (ADR-002 recommended) + Dev Tests:**

```powershell
uv sync --extra onnx --group dev
```

Verify the CLI installs cleanly with sub-300 ms cold startup:

```powershell
uv run strata --version
uv run strata --help
```

---

## Step 3: Set 8 GB RAM Ollama Environment Guardrails

Before heavy inference runs, configure Ollama to hold **at most one model** at a time with single-slot KV cache quantization so total stack RAM stays safely below the `3072 MiB` ceiling.

**Windows (PowerShell):**

```powershell
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_FLASH_ATTENTION = "1"
```

**Linux / macOS (Bash/Zsh):**

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
```

Optional Strata overrides (defaults shown):

- `STRATA_DATA_DIR=.strata` (location of `strata.db` and `models/`)
- `STRATA_PRIMARY_MODEL=qwen3:1.7b`
- `STRATA_ROUTER_MODEL=qwen3:0.6b`
- `STRATA_NUM_CTX=4096`
- `STRATA_NUM_THREAD=4`

---

## Step 4: Pull Recommended SLMs & Run Hardware Preflight (`strata doctor`)

1. Start Ollama if it is not already running, then pull the primary (`qwen3:1.7b`) and fast router (`qwen3:0.6b`) models:

```powershell
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
```

2. Run `strata doctor` to inspect Python version, physical/P/E CPU topology, system free RAM, runtime/Ollama RSS, pulled models, and Ollama environment warnings:

```powershell
uv run strata doctor
# Or machine-readable JSON:
uv run strata doctor --json
```

If Ollama runs on a non-default URL or port, pass `--host`:

```powershell
uv run strata --host http://127.0.0.1:11434 doctor
```

---

## Step 5: Fetch & Benchmark Quantized ONNX MiniLM Embeddings (Optional, ADR-002)

Strata works out of the box with zero-dependency 128-dim character n-gram vectors (`HashingNgramEmbedder`). For higher semantic cache hit rates and top-1 memory recall without loading a separate Ollama embedding model, download the quantized `Xenova/all-MiniLM-L6-v2` ONNX weights (`~22.6 MB`) into `.strata/models/all-MiniLM-L6-v2/`:

```powershell
uv sync --extra onnx
uv run strata models fetch-minilm
uv run strata bench embedders --require-onnx --save
```

Useful options:
- `--model-dir <path>` overrides `.strata/models/all-MiniLM-L6-v2`
- `--force` re-downloads existing weights
- `--json` outputs structured metrics (`benchmarks/results/embedder_ab_latest.json` when `--save` is passed)

---

## Step 6: Tune CPU Threads & Run Cached CLI Inference

1. **Find optimal `num_thread` on your CPU:**
   Hybrid CPUs (e.g., 4 P-cores + 4 E-cores) often saturate memory bandwidth before all SMT threads. Benchmark your machine:

```powershell
uv run strata bench threads --model qwen3:1.7b --threads 2,4,6,8,12 --think false --save
```

Set `$env:STRATA_NUM_THREAD` to the printed `recommended_num_thread` (typically `4` for balanced background responsiveness or `8` for peak throughput).

2. **Run a prompt through `InferenceEngine`:**
   Strata automatically applies canonical prefix packing, adaptive `think` routing, L1 (`L1_MEM` / `L1_SQLITE`) caching, and L2 semantic caching:

```powershell
# First call populates L1/L2 cache in .strata/strata.db
uv run strata run "Give three crisp benefits of SQLite WAL mode for local AI."

# Repeat call returns sub-ms cache hit from L1_SQLITE
uv run strata run "Give three crisp benefits of SQLite WAL mode for local AI."
```

Useful flags on `strata run`:
- `--think auto|false|true` — control Qwen3 reasoning trace
- `--fast` — permit `qwen3:0.6b` router model for trivial prompts
- `--no-cache` — force fresh Ollama inference
- `--max-tokens 256` — cap output length
- `--json` — return structured `InferenceResponse` JSON including cache tier, similarity, and latency telemetry

---

## Step 7: Launch & Use the OpenAI-Compatible FastAPI Server (`strata serve`)

1. Start the single-worker FastAPI gateway (optionally enabling idle-aware background memory consolidation with `--sleep-bg`):

```powershell
uv run strata serve --bind 127.0.0.1 --port 8000 --sleep-bg
```

2. Inspect gateway health & loaded models:

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s http://127.0.0.1:8000/v1/models
```

3. Send a chat completion request (supports both standard JSON and SSE `"stream": true`):

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain SQLite WAL mode in 2 sentences.\"}],\"think\":false,\"use_cache\":true}"
```

4. Store episodic memories, consolidate facts, recall knowledge, and trigger a durable agent run via REST:

```powershell
# Store an episodic turn
curl.exe -s -X POST http://127.0.0.1:8000/v1/memory/episodes `
  -H "Content-Type: application/json" `
  -d "{\"session_id\":\"s1\",\"role\":\"user\",\"content\":\"My name is Asha. I prefer concise SQL.\"}"

# Run sleep-time fact extraction & contradiction resolution
curl.exe -s -X POST http://127.0.0.1:8000/v1/memory/consolidate `
  -H "Content-Type: application/json" `
  -d "{\"batch_size\":10}"

# Query bi-temporal semantic memory (supports optional unix timestamp "as_of")
curl.exe -s -X POST http://127.0.0.1:8000/v1/memory/recall `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Where or what does Asha prefer?\",\"top_k\":5}"

# Execute a deterministic agent run
curl.exe -s -X POST http://127.0.0.1:8000/v1/runs `
  -H "Content-Type: application/json" `
  -d "{\"goal\":\"Summarize stored preferences cleanly.\",\"max_steps\":4}"
```

---

## Step 8: Use the Python SDK & Bundled Sample Apps

`StrataClient` shares the same async methods in both embedded (`local`) and HTTP (`remote`) modes.

### 8.1 Embedded Mode (`StrataClient.local`) with Typed `@tool`

```python
import asyncio
from pathlib import Path
from strata.agents.policy import Capability
from strata.config import StrataConfig
from strata.sdk import StrataClient, tool

@tool(tags=("files", "read"), capability=Capability.READ)
def read_line_count(path: str) -> int:
    """Count lines in a local file inside the workspace."""
    return len(Path(path).read_text(encoding="utf-8").splitlines())

async def main() -> None:
    cfg = StrataConfig.from_env()
    async with StrataClient.local(
        cfg,
        tools=[read_line_count],
        workspace_root=Path(".").resolve(),
    ) as client:
        # 1. Long-term cognitive memory
        await client.remember("work", "user", "My name is Asha. I moved from Pune to Bengaluru.")
        await client.consolidate()
        facts = await client.recall("Where does Asha live?")
        print("Facts:", facts)

        # 2. Cached chat inference
        chat_res = await client.chat("Summarize benefits of SQLite WAL.", think=False)
        print("Chat:", chat_res["content"], "tier:", chat_res["cache_tier"])

        # 3. Deterministic agent run with policy & journal
        run_res = await client.run_agent("Count lines in README.md and report the result.")
        print("Agent:", run_res["status"], run_res["final_answer"])

asyncio.run(main())
```

### 8.2 Remote Mode (`StrataClient.remote`) Targeting `strata serve`

```python
import asyncio
from strata.sdk import StrataClient

async def main() -> None:
    async with StrataClient.remote("http://127.0.0.1:8000") as client:
        print(await client.health())
        reply = await client.chat("What is SQLite WAL mode?", session_id="remote-1")
        print(reply["choices"][0]["message"]["content"])

asyncio.run(main())
```

### 8.3 Run the 3 Included Sample Apps

```powershell
# 1. Inspect hardware topology & ResourceGovernor headroom (works with or without Ollama running)
uv run python examples/ops_helper.py

# 2. Test CoALA bi-temporal memory supersession (Pune -> Bengaluru) & entity links
uv run python examples/memory_companion.py

# 3. Run live qwen3:1.7b workspace exploration using @tool + self-healing agent runtime
uv run python examples/workspace_analyst.py
```

---

## Step 9: Add Sandboxed JSON Plugins or MCP Tools

Create a plugin directory (e.g., `./plugins/hello.json`) declaring either a Python `entrypoint` (`module.path:callable`) or a sandboxed `command` template (executed inside the `128 MB` Win32 Job Object / timeout sandbox with NUL-byte argument rejection):

```json
{
  "name": "workspace_utils",
  "version": "0.1.0",
  "description": "Read-only helper tools",
  "tools": [
    {
      "name": "say_hello",
      "description": "Greet a named component inside a sandboxed subprocess",
      "capability": "read",
      "tags": ["demo", "greeting"],
      "always_include": false,
      "returns_untrusted": true,
      "timeout_sec": 5.0,
      "memory_limit_mb": 128,
      "parameters": {
        "type": "object",
        "properties": {
          "who": { "type": "string" }
        },
        "required": ["who"]
      },
      "command": ["python", "-c", "print('hello ' + {who!r})"]
    }
  ]
}
```

Valid `capability` values: `"read"`, `"compute"`, `"write"`, `"exec"`, `"network"`. Tools setting `"returns_untrusted": true` automatically taint context (`UNTRUSTED_DATA`) so `CaMeL` rules block unsafe follow-up `WRITE`/`EXEC`/`NETWORK` calls.

Load plugins on the server:

```powershell
uv run strata serve --plugins-dir ./plugins
```

Or adapt an external MCP tool schema in Python via [`adapt_mcp_tool`](src/strata/plugins/loader.py):

```python
from strata.agents.policy import Capability
from strata.plugins.loader import PluginRegistry, adapt_mcp_tool

reg = PluginRegistry()
reg.register_tool(
    adapt_mcp_tool(
        {"name": "fetch_issue", "description": "Read remote issue", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}},
        lambda tool_name, args: {"title": f"Issue {args['id']}"},
        capability=Capability.NETWORK,
        returns_untrusted=True,
    )
)
```

---

## Step 10: Run Automated Tests, Benchmarks & Profiling

Run the full verification suite cleanly at any time (`benchmarks/results/*.json` outputs are automatically gitignored):

```powershell
# Full unit, async, agent fault-injection, server/SDK, and evaluation suite
uv run pytest -v

# Measure CLI cold-start startup (< 300 ms SLA)
uv run strata bench cold-start --runs 5

# Run deterministic multi-pillar evaluation suite (Agent pass^3 vs naive, Memory baselines, Concurrency, Profile)
uv run strata bench eval --save

# Run full cProfile CPU hotspot + RSS + latency profile
uv run strata profile --save

# Run live Ollama caching speedup benchmark (uncached vs Turn 2 prefix vs L1 vs L2 semantic)
uv run strata bench inference --model qwen3:1.7b --save
```

---

## Troubleshooting

| Symptom | Cause & Fix |
|---|---|
| `strata doctor` reports Ollama unreachable | Ensure the Ollama daemon is running and `OLLAMA_HOST` points to `http://127.0.0.1:11434`. |
| `ollama_env_guardrails` emits `[WARN]` | Export `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, and `OLLAMA_FLASH_ATTENTION=1` in your shell/service config before starting Ollama. |
| `strata bench embedders` reports `SKIP` | Quantized MiniLM weights or `[onnx]` extra are missing. Run `uv sync --extra onnx` and `uv run strata models fetch-minilm`. Base 128-dim hashing continues working automatically. |
| `asyncio` tests fail inside restricted sandboxes | On Windows, restricted network sandboxes block loopback TCP `socketpair()` pipes required by Python 3.12 `asyncio`. Run pure synchronous tests via `uv run pytest -m "not asyncio" -v`, and run async suites in a normal terminal. |
| Slow CPU throughput on 12-thread i5 laptops | Keep `STRATA_NUM_THREAD=4` (P-cores only) or `8` (physical cores) rather than saturating all logical SMT threads. |
