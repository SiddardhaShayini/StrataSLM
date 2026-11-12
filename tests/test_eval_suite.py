"""Tests for automated reliability, memory baseline comparison, concurrency, and cProfile."""

import pytest

from strata.perf.evals import profile_hotspots, run_full_eval_suite


@pytest.mark.asyncio
async def test_full_eval_suite_passes_all_gates() -> None:
    report = await run_full_eval_suite(save=False)
    assert report.agent.passed is True
    assert report.agent.schema_validity_pct == 100.0
    assert report.agent.policy_block_pct == 100.0
    assert report.agent.pass_pow_3 is True
    assert report.agent.pass_pow_3_gain_pct > 20.0
    assert report.agent.skill_distilled_and_matched is True

    assert report.memory.passed is True
    assert report.memory.recall_at_5 == 1.0
    assert report.memory.recall_at_5 > report.memory.vector_only_acc
    assert report.memory.vector_only_acc >= report.memory.sliding_window_acc
    assert report.memory.freshness_hits == 2
    assert report.memory.abstention_correct == report.memory.abstention_cases
    assert report.memory.mean_query_latency_ms < 30.0

    assert report.inference_opt.passed is True
    assert report.inference_opt.tool_reduction_pct >= 50.0
    assert report.inference_opt.prefix_hash_stable is True
    assert report.inference_opt.prefix_reuse_pct >= 70.0

    assert report.concurrency.passed is True
    assert report.concurrency.ollama_calls_made == 1
    assert report.concurrency.coalesced_hits == 7
    assert report.concurrency.priority_preempted is True
    assert report.concurrency.red_shed_cleanly is True
    assert report.concurrency.cached_under_red_ok is True

    assert report.profile.passed is True
    assert report.profile.api_boot_ms < 800.0
    assert report.profile.l1_hit_latency_ms < 15.0
    assert report.profile.l2_hit_latency_ms < 15.0
    assert report.all_passed is True


def test_profile_hotspots_collects_cprofile_entries() -> None:
    rep = profile_hotspots(iterations=20)
    assert rep.iterations == 20
    assert rep.total_profiled_ms > 0.0
    assert len(rep.hotspots) > 0
