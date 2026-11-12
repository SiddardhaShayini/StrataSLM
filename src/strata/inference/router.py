"""Adaptive complexity & `think` mode router for 8 GB CPU-only inference."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from strata.config import StrataConfig
from strata.governor.resource_governor import GovernorState

_COMPLEX_RE = re.compile(
    r"\b("
    r"plan|architect|design|debug|diagnose|refactor|compare|analyze|analyse|"
    r"multi[- ]?step|root cause|trade[- ]?off|prove|verify|recover|why did|"
    r"step[- ]by[- ]step|algorithm|implement|optimize"
    r")\b",
    re.IGNORECASE,
)

_SIMPLE_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|yes|no|what is|define|list|summarize)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RouteDecision:
    model: str
    think: bool
    num_ctx: int
    tier: str  # "fast" | "balanced" | "deep"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveRouter:
    """Zero-overhead heuristic router balancing latency, accuracy, and 8 GB safety."""

    def __init__(self, config: StrataConfig) -> None:
        self._config = config

    def route(
        self,
        query: str,
        *,
        tool_count: int = 0,
        history_turns: int = 0,
        is_recovery: bool = False,
        governor_state: GovernorState = GovernorState.GREEN,
        prefer_fast_model: bool = False,
    ) -> RouteDecision:
        """Select model tag, `think` flag, and `num_ctx` before calling Ollama."""
        text = query.strip()
        words = len(text.split())
        complex_hits = len(_COMPLEX_RE.findall(text))
        has_code_or_json = ("```" in text) or ("{" in text and "}" in text)

        if is_recovery:
            return RouteDecision(
                model=self._config.primary_model,
                think=True,
                num_ctx=self._config.num_ctx,
                tier="deep",
                reason="Failure recovery route enables qwen3 think=True for repair/planning.",
            )

        # Under YELLOW/RED pressure, favor speed & smaller KV window
        if governor_state in {GovernorState.YELLOW, GovernorState.RED}:
            return RouteDecision(
                model=(
                    self._config.router_model
                    if prefer_fast_model
                    else self._config.primary_model
                ),
                think=False,
                num_ctx=min(2048, self._config.num_ctx),
                tier="fast",
                reason=f"Governor={governor_state.value.upper()}: forcing think=False and num_ctx<=2048.",
            )

        # Deep reasoning route only when complexity signals justify think=True latency
        if complex_hits >= 2 or (complex_hits >= 1 and (words >= 28 or has_code_or_json)):
            return RouteDecision(
                model=self._config.primary_model,
                think=True,
                num_ctx=self._config.num_ctx,
                tier="deep",
                reason=f"Complex query ({complex_hits} reasoning cues, {words} words): think=True.",
            )

        # Ultra-fast route for tiny standalone classification / greeting / short extraction
        if (
            prefer_fast_model
            and words <= 18
            and tool_count == 0
            and history_turns == 0
            and complex_hits == 0
            and _SIMPLE_RE.search(text) is not None
        ):
            return RouteDecision(
                model=self._config.router_model,
                think=False,
                num_ctx=min(2048, self._config.num_ctx),
                tier="fast",
                reason="Short standalone query eligible for fast router model (think=False).",
            )

        fast_ctx = min(2048, self._config.num_ctx) if (words <= 24 and history_turns <= 2 and tool_count <= 2) else self._config.num_ctx
        return RouteDecision(
            model=self._config.primary_model,
            think=False,
            num_ctx=fast_ctx,
            tier="balanced",
            reason=f"Standard primary route with think=False and num_ctx={fast_ctx}.",
        )
