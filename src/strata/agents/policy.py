"""Pre-action policy interceptor (AgentSpec) and taint tracking (CaMeL)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class TrustLevel(StrEnum):
    TRUSTED_SYSTEM = "trusted_system"
    TRUSTED_USER = "trusted_user"
    UNTRUSTED_DATA = "untrusted_data"


class Capability(StrEnum):
    READ = "read"
    COMPUTE = "compute"
    WRITE = "write"
    EXEC = "exec"
    NETWORK = "network"


PRIVILEGED_CAPABILITIES = frozenset(
    {Capability.WRITE, Capability.EXEC, Capability.NETWORK}
)


class EnforcementAction(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(slots=True)
class TaintedValue:
    """Tracks provenance/taint of data flowing between tools and agent actions."""

    value: Any
    trust: TrustLevel
    source: str

    @property
    def is_untrusted(self) -> bool:
        return self.trust == TrustLevel.UNTRUSTED_DATA


@dataclass(slots=True)
class ActionContext:
    tool_name: str
    capability: Capability
    arguments: Mapping[str, Any]
    context_trust: TrustLevel = TrustLevel.TRUSTED_USER
    workspace_root: Path | None = None
    approved_by_user: bool = False


@dataclass(slots=True)
class PolicyDecision:
    action: EnforcementAction
    rule_id: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.action == EnforcementAction.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
        }


@dataclass(slots=True)
class PolicyRule:
    """AgentSpec-style Trigger -> Predicate -> Enforcement rule."""

    rule_id: str
    description: str
    predicate: Callable[[ActionContext], bool]
    enforcement: EnforcementAction


_DANGEROUS_SHELL_PATTERNS = (
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[sfq]", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bpowershell\b.*-enc", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\|\s*(sh|bash|pwsh)", re.IGNORECASE),
)

_SENSITIVE_FILENAMES = frozenset({".env", ".env.local", "id_rsa", "id_ed25519", "secrets.json"})


def _has_untrusted_taint(ctx: ActionContext) -> bool:
    if ctx.context_trust == TrustLevel.UNTRUSTED_DATA:
        return True
    for val in ctx.arguments.values():
        if isinstance(val, TaintedValue) and val.is_untrusted:
            return True
    return False


def _is_path_outside_workspace(ctx: ActionContext) -> bool:
    if ctx.workspace_root is None:
        return False
    root = ctx.workspace_root.resolve()
    for key in ("path", "file", "target", "dest", "src"):
        raw = ctx.arguments.get(key)
        if raw is None:
            continue
        raw_val = raw.value if isinstance(raw, TaintedValue) else raw
        if not isinstance(raw_val, (str, Path)):
            continue
        candidate = Path(raw_val)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if resolved != root and root not in resolved.parents:
            return True
        if resolved.name.lower() in _SENSITIVE_FILENAMES and ctx.capability in {
            Capability.READ,
            Capability.WRITE,
        }:
            return True
    return False


def _matches_dangerous_exec(ctx: ActionContext) -> bool:
    if ctx.capability != Capability.EXEC:
        return False
    for key in ("command", "cmd", "script"):
        raw = ctx.arguments.get(key)
        if raw is None:
            continue
        raw_str = str(raw.value if isinstance(raw, TaintedValue) else raw)
        if any(pat.search(raw_str) for pat in _DANGEROUS_SHELL_PATTERNS):
            return True
    return False


def default_policy_rules() -> list[PolicyRule]:
    return [
        PolicyRule(
            rule_id="dangerous_shell_command",
            description="Block destructive or encoded shell commands",
            predicate=_matches_dangerous_exec,
            enforcement=EnforcementAction.DENY,
        ),
        PolicyRule(
            rule_id="workspace_boundary_and_secrets",
            description="Block paths escaping workspace_root or targeting secret files",
            predicate=_is_path_outside_workspace,
            enforcement=EnforcementAction.DENY,
        ),
        PolicyRule(
            rule_id="camel_untrusted_privilege_guard",
            description="Block untrusted tainted data from invoking privileged capabilities without approval",
            predicate=lambda ctx: (
                ctx.capability in PRIVILEGED_CAPABILITIES
                and _has_untrusted_taint(ctx)
                and not ctx.approved_by_user
            ),
            enforcement=EnforcementAction.REQUIRE_APPROVAL,
        ),
    ]


class PolicyInterceptor:
    """Synchronous pre-action guardrail evaluated before every tool execution."""

    def __init__(self, rules: Sequence[PolicyRule] | None = None) -> None:
        self._rules = list(rules) if rules is not None else default_policy_rules()

    def evaluate(self, ctx: ActionContext) -> PolicyDecision:
        for rule in self._rules:
            if rule.predicate(ctx):
                return PolicyDecision(
                    action=rule.enforcement,
                    rule_id=rule.rule_id,
                    reason=rule.description,
                )
        return PolicyDecision(
            action=EnforcementAction.ALLOW,
            rule_id="default_allow",
            reason="Passed all pre-action policy and taint checks.",
        )

    @staticmethod
    def unwrap_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Strip TaintedValue wrappers after policy approval before tool invocation."""
        return {
            k: (v.value if isinstance(v, TaintedValue) else v)
            for k, v in arguments.items()
        }
