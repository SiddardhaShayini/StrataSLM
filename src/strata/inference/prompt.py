"""Prefix-canonical prompt layout and bounded KV context packer."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from strata.inference.tool_pruner import ToolSpec

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def canonicalize_text(text: str) -> str:
    """Normalize line endings and redundant whitespace without altering semantics."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [_MULTI_SPACE_RE.sub(" ", line).rstrip() for line in normalized.split("\n")]
    return _MULTI_NL_RE.sub("\n\n", "\n".join(lines))


def estimate_tokens(text: str) -> int:
    """Zero-dependency fast token count estimator (~3.7 chars/token conservative)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3.7))


def estimate_messages_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """Estimate total prompt tokens including chat template framing overhead."""
    total = 0
    for msg in messages:
        total += 6  # role + chat framing overhead
        total += estimate_tokens(str(msg.get("content") or ""))
    return total + 4


def compact_observation(text: str, *, max_tokens: int = 180) -> tuple[str, bool]:
    """Compress verbose tool output by keeping head + tail and dropping middle bulk."""
    est = estimate_tokens(text)
    if est <= max_tokens:
        return text, False

    max_chars = max(80, int(max_tokens * 3.5))
    head_chars = int(max_chars * 0.55)
    tail_chars = max_chars - head_chars
    omitted_chars = len(text) - head_chars - tail_chars
    if omitted_chars <= 40:
        return text, False

    marker = f"\n...[truncated {omitted_chars} chars for 8GB KV budget]...\n"
    compacted = f"{text[:head_chars].rstrip()}{marker}{text[-tail_chars:].lstrip()}"
    return compacted, True


@dataclass(slots=True)
class ContextBudget:
    """Fixed token slice allocation inside num_ctx=4096 (or 2048 fast mode)."""

    num_ctx: int = 4096
    reserved_output_tokens: int = 512
    system_prefix_max_tokens: int = 700
    tools_max_tokens: int = 700
    memory_max_tokens: int = 550
    observation_max_tokens: int = 180

    @property
    def prompt_budget_tokens(self) -> int:
        return max(256, self.num_ctx - self.reserved_output_tokens)


@dataclass(slots=True)
class PackedPrompt:
    messages: list[dict[str, str]]
    prefix_hash: str
    estimated_tokens: int
    slice_tokens: dict[str, int]
    dropped_turns: int
    compacted_observations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CanonicalPromptBuilder:
    """Assembles prompts so immutable static tokens always sit at index 0..k.

    Layout order for maximum Ollama KV prefix cache reuse:
      1. Static canonical system instructions
      2. Deterministically sorted compact tool signatures
      3. Retrieved long-term memory facts (bounded slice)
      4. Recent episodic turns (newest prioritized; old tool outputs compacted)
      5. Current user query
    """

    def __init__(self, system_prompt: str, budget: ContextBudget | None = None) -> None:
        self.base_system_prompt = canonicalize_text(system_prompt)
        self.budget = budget or ContextBudget()

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        if estimate_tokens(text) <= max_tokens:
            return text
        char_cap = max(40, int(max_tokens * 3.5))
        return text[:char_cap].rstrip() + "..."

    def _render_tools_block(self, tools: Sequence[ToolSpec]) -> str:
        if not tools:
            return ""
        sorted_tools = sorted(tools, key=lambda t: t.name)
        lines = ["## Available Tools"]
        for tool in sorted_tools:
            lines.append(f"- {tool.compact_signature()}")
        return self._truncate_to_tokens("\n".join(lines), self.budget.tools_max_tokens)

    def _render_memory_block(self, memory_facts: Sequence[str]) -> str:
        if not memory_facts:
            return ""
        lines = ["## Retrieved Memory"]
        used = estimate_tokens(lines[0])
        for fact in memory_facts:
            clean = canonicalize_text(fact)
            if not clean:
                continue
            item = f"- {clean}"
            cost = estimate_tokens(item)
            if used + cost > self.budget.memory_max_tokens:
                break
            lines.append(item)
            used += cost
        return "\n".join(lines) if len(lines) > 1 else ""

    def pack(
        self,
        user_query: str,
        *,
        tools: Sequence[ToolSpec] = (),
        memory_facts: Sequence[str] = (),
        history: Sequence[dict[str, str]] = (),
    ) -> PackedPrompt:
        base_sys = self._truncate_to_tokens(
            self.base_system_prompt, self.budget.system_prefix_max_tokens
        )
        tools_block = self._render_tools_block(tools)
        static_prefix = f"{base_sys}\n\n{tools_block}" if tools_block else base_sys
        prefix_hash = hashlib.sha256(static_prefix.encode("utf-8")).hexdigest()[:16]

        memory_block = self._render_memory_block(memory_facts)
        system_content = (
            f"{static_prefix}\n\n{memory_block}" if memory_block else static_prefix
        )
        clean_query = canonicalize_text(user_query)

        # Compact tool observations inside history and fit from newest -> oldest
        compacted_obs = 0
        prepared_history: list[dict[str, str]] = []
        for turn in history:
            role = str(turn.get("role") or "user")
            content = str(turn.get("content") or "")
            if role == "tool":
                content, was_compacted = compact_observation(
                    content, max_tokens=self.budget.observation_max_tokens
                )
                if was_compacted:
                    compacted_obs += 1
            prepared_history.append({"role": role, "content": content})

        fixed_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": clean_query},
        ]
        remaining_budget = max(
            0, self.budget.prompt_budget_tokens - estimate_messages_tokens(fixed_messages)
        )

        kept_reversed: list[dict[str, str]] = []
        used_history_tokens = 0
        dropped_turns = 0
        for turn in reversed(prepared_history):
            turn_tokens = estimate_tokens(turn["content"]) + 6
            if used_history_tokens + turn_tokens <= remaining_budget:
                kept_reversed.append(turn)
                used_history_tokens += turn_tokens
            else:
                dropped_turns += 1

        kept_history = list(reversed(kept_reversed))
        messages = [
            {"role": "system", "content": system_content},
            *kept_history,
            {"role": "user", "content": clean_query},
        ]

        return PackedPrompt(
            messages=messages,
            prefix_hash=prefix_hash,
            estimated_tokens=estimate_messages_tokens(messages),
            slice_tokens={
                "system_base": estimate_tokens(base_sys),
                "tools": estimate_tokens(tools_block),
                "memory": estimate_tokens(memory_block),
                "history": used_history_tokens,
                "user": estimate_tokens(clean_query),
            },
            dropped_turns=dropped_turns,
            compacted_observations=compacted_obs,
        )
