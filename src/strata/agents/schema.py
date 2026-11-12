"""Zero-LLM JSON sanitizer + max-2 targeted Pydantic schema repair loop."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks before parsing JSON."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_balanced_json(text: str) -> str | None:
    """Find the first balanced `{...}` or `[...]` block while respecting strings."""
    start_idx = -1
    open_ch = ""
    close_ch = ""
    for i, ch in enumerate(text):
        if ch == "{":
            start_idx, open_ch, close_ch = i, "{", "}"
            break
        if ch == "[":
            start_idx, open_ch, close_ch = i, "[", "]"
            break
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for j in range(start_idx, len(text)):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start_idx : j + 1]
    return None


def extract_json_candidate(raw: str) -> str:
    """Extract JSON payload from prose, markdown fences, or <think> tags."""
    cleaned = strip_think_tags(raw)
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        fenced = fence_match.group(1).strip()
        balanced = _extract_balanced_json(fenced)
        return balanced if balanced is not None else fenced

    balanced = _extract_balanced_json(cleaned)
    return balanced if balanced is not None else cleaned


def sanitize_json_text(candidate: str) -> str:
    """Apply deterministic JSON syntax fixes (trailing commas, Python literals/dicts)."""
    fixed = _TRAILING_COMMA_RE.sub(r"\1", candidate.strip())
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass

    # Safely evaluate Python-dict style single-quoted output / True / False / None
    try:
        parsed = ast.literal_eval(fixed)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed)
    except (ValueError, SyntaxError):
        pass

    return fixed


def try_parse_model(raw: str, model_cls: type[T]) -> tuple[T | None, bool, str | None]:
    """Attempt deterministic extraction + sanitization + Pydantic validation."""
    candidate = extract_json_candidate(raw)
    # 1. Try strict JSON first
    try:
        data = json.loads(candidate)
        return model_cls.model_validate(data), False, None
    except (json.JSONDecodeError, ValidationError):
        pass

    # 2. Try deterministic sanitization
    sanitized = sanitize_json_text(candidate)
    try:
        data = json.loads(sanitized)
        return model_cls.model_validate(data), True, None
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, False, f"{type(exc).__name__}: {exc}"


@dataclass(slots=True)
class SchemaValidationResult(Generic[T]):
    ok: bool
    value: T | None
    attempts: int
    repaired_deterministically: bool
    used_llm_repair: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value.model_dump() if self.value is not None else None,
            "attempts": self.attempts,
            "repaired_deterministically": self.repaired_deterministically,
            "used_llm_repair": self.used_llm_repair,
            "error": self.error,
        }


RepairCallback = Callable[[str, str, dict[str, Any]], Awaitable[str]]


async def validate_or_repair(
    raw: str,
    model_cls: type[T],
    *,
    repair_fn: RepairCallback | None = None,
    max_repairs: int = 2,
) -> SchemaValidationResult[T]:
    """Validate SLM output with zero-LLM sanitizer, then up to `max_repairs` LLM repairs."""
    val, det_repaired, err = try_parse_model(raw, model_cls)
    if val is not None:
        return SchemaValidationResult(
            ok=True,
            value=val,
            attempts=1,
            repaired_deterministically=det_repaired,
            used_llm_repair=False,
        )

    if repair_fn is None or max_repairs <= 0:
        return SchemaValidationResult(
            ok=False,
            value=None,
            attempts=1,
            repaired_deterministically=False,
            used_llm_repair=False,
            error=err,
        )

    schema_json = model_cls.model_json_schema()
    current_raw = raw
    last_err = err or "Validation failed"
    for attempt in range(1, max_repairs + 1):
        repaired_raw = await repair_fn(current_raw, last_err, schema_json)
        val, det_repaired, err = try_parse_model(repaired_raw, model_cls)
        if val is not None:
            return SchemaValidationResult(
                ok=True,
                value=val,
                attempts=1 + attempt,
                repaired_deterministically=det_repaired,
                used_llm_repair=True,
            )
        current_raw = repaired_raw
        last_err = err or "Validation failed"

    return SchemaValidationResult(
        ok=False,
        value=None,
        attempts=1 + max_repairs,
        repaired_deterministically=False,
        used_llm_repair=True,
        error=last_err,
    )
