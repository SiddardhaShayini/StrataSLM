"""Unit tests for Pillar 2 deterministic JSON sanitizer, repair loop, and CaMeL/AgentSpec policy."""

from pathlib import Path

from pydantic import BaseModel

from strata.agents.policy import (
    ActionContext,
    Capability,
    EnforcementAction,
    PolicyInterceptor,
    TaintedValue,
    TrustLevel,
)
from strata.agents.schema import try_parse_model, validate_or_repair


class ToolPlan(BaseModel):
    action: str
    target: str
    safe: bool


def test_zero_llm_json_sanitization() -> None:
    # 1. <think> block + markdown fence + trailing comma
    raw = """<think>Let me output json</think>
Here is the plan:
```json
{
  "action": "read",
  "target": "README.md",
  "safe": true,
}
```
Hope that helps!"""
    val, det_repaired, err = try_parse_model(raw, ToolPlan)
    assert err is None
    assert det_repaired is True
    assert val == ToolPlan(action="read", target="README.md", safe=True)

    # 2. Python dict literal style
    py_raw = "{'action': 'list', 'target': '.', 'safe': True}"
    val2, det2, err2 = try_parse_model(py_raw, ToolPlan)
    assert err2 is None
    assert det2 is True
    assert val2 == ToolPlan(action="list", target=".", safe=True)


async def test_targeted_repair_loop_succeeds_within_two_attempts() -> None:
    calls = 0

    async def fake_repair(prev: str, error: str, schema: dict) -> str:
        nonlocal calls
        calls += 1
        assert "ToolPlan" in str(schema) or "properties" in schema
        if calls == 1:
            return '{"action": "read", "target": "a.py"}'  # still missing `safe`
        return '{"action": "read", "target": "a.py", "safe": true}'

    res = await validate_or_repair("not json at all", ToolPlan, repair_fn=fake_repair, max_repairs=2)
    assert res.ok is True
    assert res.used_llm_repair is True
    assert res.attempts == 3
    assert res.value == ToolPlan(action="read", target="a.py", safe=True)


def test_policy_interceptor_blocks_injection_traversal_and_dangerous_exec(tmp_path: Path) -> None:
    policy = PolicyInterceptor()

    # 1. Untrusted web/file content trying to trigger EXEC -> REQUIRE_APPROVAL (CaMeL)
    tainted_ctx = ActionContext(
        tool_name="run_shell",
        capability=Capability.EXEC,
        arguments={"command": TaintedValue("echo hi", TrustLevel.UNTRUSTED_DATA, "web_page")},
        workspace_root=tmp_path,
    )
    dec1 = policy.evaluate(tainted_ctx)
    assert dec1.action == EnforcementAction.REQUIRE_APPROVAL
    assert dec1.rule_id == "camel_untrusted_privilege_guard"

    # Untrusted data reading safe file inside workspace -> ALLOW
    tainted_read = ActionContext(
        tool_name="read_file",
        capability=Capability.READ,
        arguments={"path": "notes.txt"},
        context_trust=TrustLevel.UNTRUSTED_DATA,
        workspace_root=tmp_path,
    )
    assert policy.evaluate(tainted_read).allowed is True

    # 2. Path traversal outside workspace -> DENY
    escape = ActionContext(
        tool_name="read_file",
        capability=Capability.READ,
        arguments={"path": "../../Windows/System32/drivers/etc/hosts"},
        workspace_root=tmp_path,
    )
    assert policy.evaluate(escape).action == EnforcementAction.DENY

    # 3. Secret file access -> DENY
    secret = ActionContext(
        tool_name="read_file",
        capability=Capability.READ,
        arguments={"path": ".env"},
        workspace_root=tmp_path,
    )
    assert policy.evaluate(secret).action == EnforcementAction.DENY

    # 4. Destructive shell command -> DENY
    rm_rf = ActionContext(
        tool_name="run_shell",
        capability=Capability.EXEC,
        arguments={"command": "rm -rf /"},
        workspace_root=tmp_path,
    )
    assert policy.evaluate(rm_rf).action == EnforcementAction.DENY
