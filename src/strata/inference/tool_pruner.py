"""Zero-LLM dynamic top-k tool schema pruner with canonical ordering for KV reuse."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    # Split camelCase/snake_case smoothly into lowercase alphanumeric tokens
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return _TOKEN_RE.findall(expanded.lower())


@dataclass(slots=True)
class ToolSpec:
    """Canonical tool metadata used for pruning, prompt injection, and policy checks."""

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    tags: tuple[str, ...] = ()
    always_include: bool = False
    capability: str = "read"

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def compact_signature(self) -> str:
        """Compact single-line tool signature to minimize SLM prefill tokens."""
        props = self.parameters.get("properties") or {}
        req = set(self.parameters.get("required") or [])
        args: list[str] = []
        for k in sorted(props.keys()):
            typ = (props[k] or {}).get("type", "any")
            mark = "*" if k in req else "?"
            args.append(f"{k}{mark}:{typ}")
        return f"{self.name}({', '.join(args)}) — {self.description.strip()}"


@dataclass(slots=True)
class PrunedToolsResult:
    selected: list[ToolSpec]
    schema_hash: str
    total_tools: int
    selected_count: int
    estimated_full_chars: int
    estimated_pruned_chars: int

    @property
    def char_reduction_pct(self) -> float:
        if self.estimated_full_chars <= 0:
            return 0.0
        saved = self.estimated_full_chars - self.estimated_pruned_chars
        return round(max(0.0, saved) / self.estimated_full_chars * 100.0, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_names": [t.name for t in self.selected],
            "schema_hash": self.schema_hash,
            "total_tools": self.total_tools,
            "selected_count": self.selected_count,
            "estimated_full_chars": self.estimated_full_chars,
            "estimated_pruned_chars": self.estimated_pruned_chars,
            "char_reduction_pct": self.char_reduction_pct,
        }


def compute_schema_hash(tools: Sequence[ToolSpec]) -> str:
    """Stable hash of sorted tool definitions so cache/KV keys invalidate on edits."""
    canonical = [
        t.to_ollama_tool()
        for t in sorted(tools, key=lambda item: item.name)
    ]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ToolPruner:
    """Fast BM25 + exact-mention tool selector preserving canonical alphabetical order."""

    def __init__(self, tools: Sequence[ToolSpec]) -> None:
        self._tools = list(tools)
        self._doc_tokens: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 1.0
        self._build_index()

    def _build_index(self) -> None:
        if not self._tools:
            return
        total_len = 0
        for tool in self._tools:
            props = tool.parameters.get("properties") or {}
            prop_text = " ".join(
                f"{k} {(v or {}).get('description', '')}" for k, v in props.items()
            )
            full_text = f"{tool.name} {tool.name} {' '.join(tool.tags)} {tool.description} {prop_text}"
            toks = _tokenize(full_text)
            counts = Counter(toks)
            self._doc_tokens.append(counts)
            total_len += sum(counts.values())
            for term in counts.keys():
                self._df[term] += 1
        self._avg_len = max(1.0, total_len / len(self._tools))

    def select(self, query: str, *, top_k: int = 4) -> PrunedToolsResult:
        """Select up to `top_k` relevant tools (+ pinned tools), sorted by name."""
        if not self._tools:
            return PrunedToolsResult(
                selected=[],
                schema_hash=compute_schema_hash([]),
                total_tools=0,
                selected_count=0,
                estimated_full_chars=0,
                estimated_pruned_chars=0,
            )

        q_tokens = _tokenize(query)
        q_lower = query.lower()
        n_docs = len(self._tools)
        k1 = 1.5
        b = 0.75

        scored: list[tuple[float, ToolSpec]] = []
        for idx, tool in enumerate(self._tools):
            if tool.always_include:
                continue
            tf_map = self._doc_tokens[idx]
            doc_len = sum(tf_map.values()) or 1
            score = 0.0
            for term in q_tokens:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + k1 * (1.0 - b + b * (doc_len / self._avg_len))
                score += idf * ((tf * (k1 + 1.0)) / denom)

            # Strong boost for explicit tool-name or tag mentions
            if tool.name.lower() in q_lower:
                score += 10.0
            for tag in tool.tags:
                if tag.lower() in q_lower:
                    score += 2.0

            scored.append((score, tool))

        pinned = [t for t in self._tools if t.always_include]
        remaining_slots = max(0, top_k - len(pinned))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        chosen = list(pinned)
        for score, tool in scored[:remaining_slots]:
            if score > 0 or len(chosen) < min(top_k, len(self._tools)):
                chosen.append(tool)

        # Canonical alphabetical ordering preserves KV prefix stability across turns!
        chosen.sort(key=lambda t: t.name)

        full_chars = sum(len(json.dumps(t.to_ollama_tool())) for t in self._tools)
        pruned_chars = sum(len(json.dumps(t.to_ollama_tool())) for t in chosen)
        return PrunedToolsResult(
            selected=chosen,
            schema_hash=compute_schema_hash(chosen),
            total_tools=len(self._tools),
            selected_count=len(chosen),
            estimated_full_chars=full_chars,
            estimated_pruned_chars=pruned_chars,
        )
