"""StrataSLM CLI entry point with lazy imports for <300 ms cold startup."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strata",
        description="StrataSLM: Featherweight local AI runtime & cognitive memory engine",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama base URL (default: OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doc_p = sub.add_parser(
        "doctor",
        help="Inspect hardware topology, 8 GB RAM headroom, Ollama env, and models",
    )
    doc_p.add_argument("--json", action="store_true", help="Emit JSON output")

    bench_p = sub.add_parser("bench", help="Run hardware and inference benchmarks")
    bench_sub = bench_p.add_subparsers(dest="bench_command", required=True)

    threads_p = bench_sub.add_parser(
        "threads",
        help="Sweep Ollama num_thread values for TTFT, tok/s, and RSS",
    )
    threads_p.add_argument("--model", default=None, help="Target Ollama model tag")
    threads_p.add_argument(
        "--threads",
        default="2,4,6,8,12",
        help="Comma-separated num_thread values (default: 2,4,6,8,12)",
    )
    threads_p.add_argument("--num-ctx", type=int, default=4096, help="Context window")
    threads_p.add_argument(
        "--think",
        choices=("false", "true"),
        default="false",
        help="Qwen3 thinking toggle (default: false)",
    )
    threads_p.add_argument("--trials", type=int, default=2, help="Warm trials per thread count")
    threads_p.add_argument("--max-tokens", type=int, default=128, help="Max predicted tokens")
    threads_p.add_argument("--save", action="store_true", help="Save JSON report to benchmarks/results/")
    threads_p.add_argument("--json", action="store_true", help="Emit JSON output")

    cold_p = bench_sub.add_parser(
        "cold-start",
        help="Measure CLI lazy-import startup latency against <300 ms target",
    )
    cold_p.add_argument("--runs", type=int, default=5, help="Subprocess runs")
    cold_p.add_argument("--json", action="store_true", help="Emit JSON output")

    eval_p = bench_sub.add_parser(
        "eval",
        help="Run agent reliability (pass^3, injection), memory, and API/cache evaluations",
    )
    eval_p.add_argument("--save", action="store_true", help="Save JSON report to benchmarks/results/")
    eval_p.add_argument("--json", action="store_true", help="Emit JSON output")

    inf_p = bench_sub.add_parser(
        "inference",
        help="Benchmark live Ollama uncached vs prefix-warm vs L1/L2 cache speedups",
    )
    inf_p.add_argument("--model", default=None, help="Target Ollama model tag")
    inf_p.add_argument("--save", action="store_true", help="Save JSON report to benchmarks/results/")
    inf_p.add_argument("--json", action="store_true", help="Emit JSON output")

    emb_p = bench_sub.add_parser(
        "embedders",
        help="A/B benchmark HashingNgramEmbedder(128) vs OnnxMiniLMEmbedder(384)",
    )
    emb_p.add_argument(
        "--model-dir",
        default=".strata/models/all-MiniLM-L6-v2",
        help="Directory holding MiniLM ONNX model and tokenizer.json",
    )
    emb_p.add_argument(
        "--require-onnx",
        action="store_true",
        help="Fail if ONNX MiniLM weights/runtime are unavailable",
    )
    emb_p.add_argument("--save", action="store_true", help="Save JSON report to benchmarks/results/")
    emb_p.add_argument("--json", action="store_true", help="Emit JSON output")

    serve_p = sub.add_parser(
        "serve",
        help="Start single-worker OpenAI-compatible FastAPI gateway (8 GB safe)",
    )
    serve_p.add_argument("--bind", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    serve_p.add_argument("--plugins-dir", default=None, help="Optional JSON plugin manifest directory")
    serve_p.add_argument("--sleep-bg", action="store_true", help="Enable background sleep-time consolidator")

    profile_p = sub.add_parser(
        "profile",
        help="Run cold-start, API boot, cache hit, memory latency, cProfile hotspots, and RSS check",
    )
    profile_p.add_argument("--save", action="store_true", help="Save JSON report to benchmarks/results/")
    profile_p.add_argument("--json", action="store_true", help="Emit JSON output")

    run_p = sub.add_parser(
        "run",
        help="Run a prompt through StrataSLM InferenceEngine (prefix packer, router, L1 cache)",
    )
    run_p.add_argument("prompt", help="User prompt text")
    run_p.add_argument("--model", default=None, help="Override routed model tag")
    run_p.add_argument(
        "--think",
        choices=("auto", "false", "true"),
        default="auto",
        help="Override thinking mode (default: auto)",
    )
    run_p.add_argument("--no-cache", action="store_true", help="Bypass L1 cache")
    run_p.add_argument("--fast", action="store_true", help="Allow router_model for simple queries")
    run_p.add_argument("--max-tokens", type=int, default=256, help="Max predicted tokens")
    run_p.add_argument("--json", action="store_true", help="Emit JSON output")

    models_p = sub.add_parser("models", help="Manage local optional model weights")
    models_sub = models_p.add_subparsers(dest="models_command", required=True)
    fetch_p = models_sub.add_parser(
        "fetch-minilm",
        help="Download quantized all-MiniLM-L6-v2 ONNX weights into .strata/models/",
    )
    fetch_p.add_argument(
        "--model-dir",
        default=".strata/models/all-MiniLM-L6-v2",
        help="Target directory for model_quantized.onnx and tokenizer.json",
    )
    fetch_p.add_argument("--force", action="store_true", help="Re-download existing files")
    fetch_p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    fetch_p.add_argument("--json", action="store_true", help="Emit JSON output")

    return parser


def _cmd_doctor(host: str | None, as_json: bool) -> int:
    from strata.config import StrataConfig
    from strata.perf.bench import run_doctor

    config = StrataConfig.from_env(ollama_host=host)
    report = asyncio.run(run_doctor(config))

    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.passed else 1

    print("=== StrataSLM Doctor ===")
    print(f"Timestamp : {report.timestamp}")
    print(
        f"CPU       : {report.cpu.physical_cores} physical / {report.cpu.logical_cpus} logical"
        + (
            f" ({report.cpu.p_cores}P + {report.cpu.e_cores}E)"
            if report.cpu.is_hybrid
            else ""
        )
        + f" | recommended num_thread={report.cpu.recommended_num_thread}"
    )
    mem = report.governor.memory
    print(
        f"Memory    : avail {mem.available_mb:.0f} MiB / {mem.total_mb:.0f} MiB "
        f"| Strata RSS {mem.strata_rss_mb:.1f} MiB | Ollama RSS {mem.ollama_rss_mb:.1f} MiB"
    )
    print(
        f"Governor  : {report.governor.state.value.upper()} "
        f"(CPU {report.governor.cpu_percent:.1f}%, Disk free {report.disk_free_gb:.1f} GiB)"
    )
    print("\nChecks:")
    for chk in report.checks:
        print(f"  [{chk.status:<4}] {chk.name:<22} {chk.detail}")

    if report.ollama.installed_models:
        print("\nInstalled Ollama Models:")
        for m in report.ollama.installed_models:
            extra = f"{m.parameter_size or '?'} {m.quantization or ''}".strip()
            print(f"  - {m.name:<48} {m.size_gb:>5.2f} GiB  ({extra})")

    return 0 if report.passed else 1


def _cmd_bench_threads(args: argparse.Namespace) -> int:
    from strata.config import StrataConfig
    from strata.perf.bench import run_thread_sweep

    config = StrataConfig.from_env(ollama_host=args.host)
    thread_list = [int(part.strip()) for part in args.threads.split(",") if part.strip()]
    think_flag = args.think == "true"

    report = asyncio.run(
        run_thread_sweep(
            config,
            model=args.model,
            threads=thread_list,
            num_ctx=args.num_ctx,
            think=think_flag,
            trials=args.trials,
            max_tokens=args.max_tokens,
            save=args.save,
            results_dir=Path("benchmarks/results"),
        )
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    print(
        f"=== Thread Sweep: {report.model} "
        f"(think={report.think}, num_ctx={report.num_ctx}, max_tokens={report.max_tokens}) ==="
    )
    header = (
        f"{'Threads':>7} | {'Cold TTFT':>10} | {'Warm TTFT':>10} | "
        f"{'Prompt t/s':>10} | {'Eval t/s':>9} | {'Ollama RSS':>11} | {'Stack RSS':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in report.results:
        print(
            f"{r.num_thread:>7} | {r.cold_ttft_ms:>8.1f}ms | {r.warm_ttft_mean_ms:>8.1f}ms | "
            f"{r.warm_prompt_tok_s:>10.2f} | {r.warm_eval_tok_s:>9.2f} | "
            f"{r.peak_ollama_rss_mb:>9.1f}MB | {r.peak_stack_rss_mb:>8.1f}MB"
        )
    print(f"\nRecommended num_thread: {report.recommended_num_thread}")
    if report.saved_to:
        print(f"Saved report: {report.saved_to}")
    return 0


def _cmd_bench_cold_start(runs: int, as_json: bool) -> int:
    from strata.perf.bench import measure_cold_start

    report = measure_cold_start(runs=runs)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.passed else 1

    status = "PASS" if report.passed else "FAIL"
    print(
        f"[{status}] CLI cold-start ({report.runs} runs): "
        f"min={report.min_ms:.1f}ms, median={report.median_ms:.1f}ms, "
        f"max={report.max_ms:.1f}ms (target < {report.target_ms:.0f}ms)"
    )
    return 0 if report.passed else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from strata.config import StrataConfig
    from strata.inference.engine import InferenceEngine

    config = StrataConfig.from_env(ollama_host=args.host)
    force_think: bool | None = None
    if args.think == "true":
        force_think = True
    elif args.think == "false":
        force_think = False

    async def _exec():
        async with InferenceEngine(config) as engine:
            return await engine.complete(
                args.prompt,
                force_model=args.model,
                force_think=force_think,
                use_cache=not args.no_cache,
                prefer_fast_model=args.fast,
                max_tokens=args.max_tokens,
            )

    res = asyncio.run(_exec())
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    print(res.content)
    if res.cached:
        print(
            f"\n[cache={res.cache_tier} {res.cache_latency_ms:.2f}ms | "
            f"model={res.model} think={res.think} ctx={res.num_ctx}]"
        )
    elif res.metrics:
        print(
            f"\n[uncached | model={res.model} think={res.think} ctx={res.num_ctx} | "
            f"ttft={res.metrics.ttft_ms:.1f}ms eval={res.metrics.eval_tok_per_sec:.1f}t/s "
            f"wall={res.metrics.wall_ms:.1f}ms]"
        )
    return 0


def _cmd_bench_eval(save: bool, as_json: bool) -> int:
    from strata.perf.evals import run_full_eval_suite

    report = asyncio.run(run_full_eval_suite(save=save))
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.all_passed else 1

    print("=== StrataSLM Evaluation Suite ===")
    print(
        f"[{'PASS' if report.agent.passed else 'FAIL'}] Agent Reliability : "
        f"JSON={report.agent.schema_validity_pct}% | "
        f"PolicyBlock={report.agent.policy_block_pct}% | "
        f"pass^3={report.agent.pass_at_3_successes}/{report.agent.pass_at_3_trials} "
        f"(vs naive {report.agent.naive_react_pass_at_3_successes}/{report.agent.pass_at_3_trials}, "
        f"+{report.agent.pass_pow_3_gain_pct}%) | skill_distilled={report.agent.skill_distilled_and_matched}"
    )
    print(
        f"[{'PASS' if report.memory.passed else 'FAIL'}] Cognitive Memory  : "
        f"Recall@5={report.memory.recall_at_5:.2f} "
        f"(vec={report.memory.vector_only_acc:.2f}, win={report.memory.sliding_window_acc:.2f}, none={report.memory.no_memory_acc:.2f}) | "
        f"Freshness={report.memory.freshness_hits}/{report.memory.freshness_cases} | "
        f"Abstention={report.memory.abstention_correct}/{report.memory.abstention_cases} | "
        f"latency={report.memory.mean_query_latency_ms:.2f}ms | "
        f"RSS={report.memory.runtime_rss_mb:.1f}MB"
    )
    print(
        f"[{'PASS' if report.inference_opt.passed else 'FAIL'}] Inference Opts    : "
        f"tool_prune=-{report.inference_opt.tool_reduction_pct}% | "
        f"prefix_reuse={report.inference_opt.prefix_reuse_pct}% "
        f"(hash_stable={report.inference_opt.prefix_hash_stable})"
    )
    print(
        f"[{'PASS' if report.concurrency.passed else 'FAIL'}] Concurrency & OOM : "
        f"coalesced={report.concurrency.coalesced_hits}/{report.concurrency.concurrent_requests - 1} | "
        f"priority_preempt={report.concurrency.priority_preempted} | "
        f"RED_shed={report.concurrency.red_shed_cleanly}"
    )
    print(
        f"[{'PASS' if report.profile.passed else 'FAIL'}] Runtime Profile   : "
        f"API boot={report.profile.api_boot_ms:.1f}ms | "
        f"L1={report.profile.l1_hit_latency_ms:.2f}ms | "
        f"L2={report.profile.l2_hit_latency_ms:.2f}ms | "
        f"RSS={report.profile.strata_rss_mb:.1f}MB"
    )
    return 0 if report.all_passed else 1


def _cmd_bench_inference(args: argparse.Namespace) -> int:
    from strata.config import StrataConfig
    from strata.perf.evals import run_live_inference_benchmark

    config = StrataConfig.from_env(ollama_host=args.host)
    report = asyncio.run(
        run_live_inference_benchmark(
            config,
            model=args.model,
            save=args.save,
            results_dir=Path("benchmarks/results"),
        )
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.within_3gb_ceiling else 1

    print(f"=== Live Inference Benchmark ({report.model}) ===")
    print(
        f"Uncached Ollama  : wall={report.uncached_wall_ms:.1f}ms | "
        f"ttft={report.uncached_ttft_ms:.1f}ms | {report.uncached_tok_s:.1f} tok/s"
    )
    print(
        f"Strata Prefix    : wall={report.strata_prefix_wall_ms:.1f}ms | "
        f"ttft={report.strata_prefix_ttft_ms:.1f}ms ({report.ttft_prefix_speedup_x:.2f}x TTFT)"
    )
    print(
        f"L1 Cache Hit     : {report.l1_hit_ms:.3f}ms ({report.l1_tier}, {report.l1_speedup_x:.1f}x faster)"
    )
    print(
        f"L2 Semantic Hit  : {report.l2_hit_ms:.3f}ms ({report.l2_tier}, "
        f"sim={report.l2_similarity}, {report.l2_speedup_x:.1f}x faster)"
    )
    print(
        f"Peak Stack RSS   : {report.peak_stack_rss_mb:.1f} MB "
        f"(within <=3072 MB ceiling: {report.within_3gb_ceiling})"
    )
    if report.saved_to:
        print(f"Saved report     : {report.saved_to}")
    return 0 if report.within_3gb_ceiling else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from strata.config import StrataConfig
    from strata.server.app import create_app

    config = StrataConfig.from_env(ollama_host=args.host)
    app = create_app(
        config,
        plugins_dir=args.plugins_dir,
        workspace_root=Path(".").resolve(),
        enable_background_sleep=args.sleep_bg,
    )
    uvicorn.run(app, host=args.bind, port=args.port, workers=1, log_level="info")
    return 0


def _cmd_profile(save: bool, as_json: bool) -> int:
    from strata.perf.bench import measure_cold_start
    from strata.perf.evals import profile_hotspots, run_full_eval_suite

    cold = measure_cold_start(runs=3)
    suite = asyncio.run(run_full_eval_suite(save=False))
    cprof = profile_hotspots(iterations=80)
    passed = cold.passed and suite.all_passed
    payload = {
        "cold_start": cold.to_dict(),
        "eval_suite": suite.to_dict(),
        "cpu_profile": cprof.to_dict(),
        "passed": passed,
    }
    if save:
        out_dir = Path("benchmarks/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "profile_latest.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0 if passed else 1

    print(
        f"[{'PASS' if cold.passed else 'FAIL'}] Cold-start median={cold.median_ms:.1f}ms (<300ms)"
    )
    print(
        f"[{'PASS' if suite.profile.passed else 'FAIL'}] API boot={suite.profile.api_boot_ms:.1f}ms (<800ms) | "
        f"L1={suite.profile.l1_hit_latency_ms:.2f}ms | L2={suite.profile.l2_hit_latency_ms:.2f}ms | "
        f"Strata RSS={suite.profile.strata_rss_mb:.1f}MB (<180MB)"
    )
    print(
        f"[{'PASS' if suite.memory.passed else 'FAIL'}] Memory query={suite.memory.mean_query_latency_ms:.2f}ms (<30ms) | "
        f"Recall@5={suite.memory.recall_at_5:.2f} (vec={suite.memory.vector_only_acc:.2f}, win={suite.memory.sliding_window_acc:.2f})"
    )
    print(f"CPU Hotspots ({cprof.iterations} iters, {cprof.total_profiled_ms:.1f}ms total):")
    for h in cprof.hotspots[:5]:
        print(
            f"  - {h.function:<24} {h.file_line:<24} calls={h.ncalls:<5} tot={h.tottime_ms:.2f}ms"
        )
    return 0 if passed else 1


def _cmd_bench_embedders(args: argparse.Namespace) -> int:
    from strata.perf.evals import eval_embedder_ab

    report = eval_embedder_ab(
        model_dir=args.model_dir,
        save=args.save,
        require_onnx=args.require_onnx,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    h = report["hashing"]
    o = report["onnx_minilm"]
    print(
        f"[HASH] dim={h['dim']} | margin={h['separation_margin']:.4f} "
        f"(para={h['mean_paraphrase_cosine']:.3f}, unrel={h['mean_unrelated_cosine']:.3f}) | "
        f"L2_hit={h['l2_paraphrase_hit_rate']:.2f} (fp={h['l2_false_positives']}) | "
        f"Mem@1={h['memory_recall_at_1']:.2f} Mem@5={h['memory_recall_at_5']:.2f} | "
        f"{h['encode_ms_per_text']:.2f}ms/txt"
    )
    if o.get("available"):
        print(
            f"[{report['status']}] ONNX MiniLM dim={o['dim']} | margin={o['separation_margin']:.4f} "
            f"(para={o['mean_paraphrase_cosine']:.3f}, unrel={o['mean_unrelated_cosine']:.3f}) | "
            f"L2_hit={o['l2_paraphrase_hit_rate']:.2f} (fp={o['l2_false_positives']}) | "
            f"Mem@1={o['memory_recall_at_1']:.2f} Mem@5={o['memory_recall_at_5']:.2f} | "
            f"{o['encode_ms_per_text']:.2f}ms/txt | "
            f"RSS={o['rss_after_mb']:.1f}MB (+{o['rss_delta_mb']:.1f}MB)"
        )
    else:
        print(f"[{report['status']}] ONNX MiniLM unavailable: {o.get('reason')}")
    if report.get("saved_to"):
        print(f"Saved report -> {report['saved_to']}")
    return 0 if report["passed"] else 1


def _cmd_models_fetch_minilm(args: argparse.Namespace) -> int:
    from strata.memory.embedder import fetch_minilm_onnx

    res = fetch_minilm_onnx(
        args.model_dir,
        force=args.force,
        timeout_sec=args.timeout,
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    mb = sum(int(v) for v in res["bytes"].values()) / (1024.0 * 1024.0)
    print(
        f"[PASS] MiniLM ONNX ready in {res['model_dir']} "
        f"(downloaded={res['downloaded']}, skipped={res['skipped']}, total={mb:.2f} MB)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _cmd_doctor(args.host, args.json)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "profile":
        return _cmd_profile(args.save, args.json)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "models":
        if args.models_command == "fetch-minilm":
            return _cmd_models_fetch_minilm(args)
    if args.command == "bench":
        if args.bench_command == "threads":
            return _cmd_bench_threads(args)
        if args.bench_command == "cold-start":
            return _cmd_bench_cold_start(args.runs, args.json)
        if args.bench_command == "eval":
            return _cmd_bench_eval(args.save, args.json)
        if args.bench_command == "inference":
            return _cmd_bench_inference(args)
        if args.bench_command == "embedders":
            return _cmd_bench_embedders(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
