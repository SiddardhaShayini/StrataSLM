"""Deterministic Agent State Machine: PLAN -> VALIDATE -> AUTHORIZE -> EXECUTE -> VERIFY -> COMMIT."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Sequence

from pydantic import BaseModel, Field

from strata.agents.journal import AgentPhase, DurableStepJournal, RunStatus
from strata.agents.policy import (
    ActionContext,
    Capability,
    EnforcementAction,
    PolicyInterceptor,
    TaintedValue,
    TrustLevel,
)
from strata.agents.schema import validate_or_repair
from strata.inference.client import RequestPriority
from strata.inference.engine import InferenceEngine
from strata.inference.tool_pruner import ToolSpec

if TYPE_CHECKING:
    from strata.memory.store import CognitiveMemoryStore

AGENT_SYSTEM_PROMPT = """You are StrataSLM Agent. Respond ONLY with a valid JSON object matching:
{
  "thought": "brief reasoning",
  "decision": "call_tool" | "final_answer",
  "tool_name": "name of tool if decision == call_tool else null",
  "arguments": { ... tool args if decision == call_tool else {} },
  "final_answer": "concise final answer if decision == final_answer else null"
}
Never invent tool names. Stop with decision="final_answer" once the goal is answered."""


class AgentActionDecision(BaseModel):
    thought: str = ""
    decision: Literal["call_tool", "final_answer"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any] | Any]
RollbackHandler = Callable[[dict[str, Any], Any], Awaitable[None] | None]


@dataclass(slots=True)
class RegisteredTool:
    spec: ToolSpec
    capability: Capability
    handler: ToolHandler
    returns_untrusted: bool = False
    compensator: RollbackHandler | None = None


@dataclass(slots=True)
class AgentRunResult:
    run_id: str
    status: RunStatus
    final_answer: str | None
    steps_executed: int
    halt_reason: str | None = None
    rolled_back_steps: int = 0
    history: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class DeterministicAgentRuntime:
    """Bounded, crash-resumable agent state machine with policy & loop guards."""

    def __init__(
        self,
        engine: InferenceEngine,
        tools: Sequence[RegisteredTool],
        *,
        policy: PolicyInterceptor | None = None,
        workspace_root: Path | None = None,
        memory: CognitiveMemoryStore | None = None,
        max_steps: int = 6,
    ) -> None:
        self.engine = engine
        self.tools = {t.spec.name: t for t in tools}
        self.tool_specs = [t.spec for t in tools]
        self.journal = DurableStepJournal(engine.db)
        self.policy = policy or PolicyInterceptor()
        self.workspace_root = workspace_root
        self.memory = memory
        self.max_steps = max_steps

    @staticmethod
    def _skill_slug(goal: str) -> str:
        words = re.findall(r"[a-z0-9]+", goal.lower())[:5]
        prefix = "_".join(words) if words else "skill"
        digest = hashlib.sha256(goal.lower().strip().encode("utf-8")).hexdigest()[:6]
        return f"skill_{prefix}_{digest}"

    @staticmethod
    def _call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"

    async def _invoke_maybe_async(self, fn: Callable[..., Any], *args: Any) -> Any:
        res = fn(*args)
        if inspect.isawaitable(res):
            return await res
        return res

    async def _rollback_committed(
        self,
        run_id: str,
        executed_actions: list[tuple[RegisteredTool, dict[str, Any], Any]],
    ) -> int:
        for tool, args, result in reversed(executed_actions):
            if tool.compensator is not None:
                await self._invoke_maybe_async(tool.compensator, args, result)
        return self.journal.mark_steps_rolled_back(run_id)

    async def run(
        self,
        goal: str,
        *,
        resume_run_id: str | None = None,
        memory_facts: Sequence[str] = (),
        rollback_on_failure: bool = True,
    ) -> AgentRunResult:
        if resume_run_id is not None:
            record = self.journal.load_run(resume_run_id)
            if record is None:
                raise ValueError(f"Unknown run_id {resume_run_id!r}")
            run_id = record.run_id
            step_idx = record.step_index
            history: list[dict[str, str]] = list(record.state.get("history") or [])
            context_trust = TrustLevel(
                record.state.get("context_trust", TrustLevel.TRUSTED_USER.value)
            )
            seen_signatures: list[str] = list(record.state.get("seen_signatures") or [])
        else:
            record = self.journal.create_run(
                goal,
                initial_state={
                    "history": [],
                    "context_trust": TrustLevel.TRUSTED_USER.value,
                    "seen_signatures": [],
                },
            )
            run_id = record.run_id
            step_idx = 0
            history = []
            context_trust = TrustLevel.TRUSTED_USER
            seen_signatures = []

        executed_for_saga: list[tuple[RegisteredTool, dict[str, Any], Any]] = []
        consecutive_errors = 0

        effective_facts = list(memory_facts)
        if self.memory is not None:
            matched_skill = self.memory.match_skill(goal)
            if matched_skill is not None and matched_skill.steps:
                seq_preview = " -> ".join(str(s.get("tool", "")) for s in matched_skill.steps)
                effective_facts.insert(
                    0,
                    f"Procedural skill '{matched_skill.name}' ({matched_skill.success_count}x): {seq_preview}",
                )

        while step_idx < self.max_steps:
            # --- PLAN ---
            prompt = (
                f"Goal: {goal}\nDecide next JSON action."
                if not history
                else f"Goal: {goal}\nReview observations and return next JSON action."
            )
            plan_res = await self.engine.complete(
                prompt,
                system_prompt=AGENT_SYSTEM_PROMPT,
                tools=self.tool_specs,
                memory_facts=effective_facts,
                history=history,
                use_cache=False,
                temperature=0.1,
                max_tokens=320,
                priority=RequestPriority.HIGH,
            )

            # --- VALIDATE (zero-LLM sanitize + max-2 targeted repair) ---
            async def _repair(prev_raw: str, err: str, schema: dict[str, Any]) -> str:
                repair_prompt = (
                    f"Fix this invalid JSON output so it matches the schema.\n"
                    f"Validation error: {err}\n"
                    f"Output ONLY valid JSON.\nMalformed output:\n{prev_raw}"
                )
                rep = await self.engine.complete(
                    repair_prompt,
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    use_cache=False,
                    force_think=True,
                    temperature=0.0,
                    max_tokens=320,
                    priority=RequestPriority.HIGH,
                )
                return rep.content

            val_res = await validate_or_repair(
                plan_res.content,
                AgentActionDecision,
                repair_fn=_repair,
                max_repairs=2,
            )
            if not val_res.ok or val_res.value is None:
                rolled_back = (
                    await self._rollback_committed(run_id, executed_for_saga)
                    if rollback_on_failure
                    else 0
                )
                self.journal.checkpoint_run(
                    run_id,
                    status=RunStatus.FAILED,
                    current_state=AgentPhase.VALIDATE,
                    step_index=step_idx,
                    state={
                        "history": history,
                        "context_trust": context_trust.value,
                        "seen_signatures": seen_signatures,
                    },
                )
                return AgentRunResult(
                    run_id=run_id,
                    status=RunStatus.FAILED,
                    final_answer=None,
                    steps_executed=step_idx,
                    halt_reason=f"Schema validation failed: {val_res.error}",
                    rolled_back_steps=rolled_back,
                    history=history,
                )

            action = val_res.value
            if action.decision == "final_answer":
                answer = action.final_answer or ""
                self.journal.record_step(
                    run_id,
                    step_index=step_idx,
                    phase=AgentPhase.COMMIT,
                    idempotency_key=f"step-{step_idx}-final",
                    action_name="final_answer",
                    input_data=action.model_dump(),
                    output_data={"final_answer": answer},
                    status="committed",
                )
                self.journal.checkpoint_run(
                    run_id,
                    status=RunStatus.COMPLETED,
                    current_state=AgentPhase.COMMIT,
                    step_index=step_idx + 1,
                    state={
                        "history": history,
                        "context_trust": context_trust.value,
                        "seen_signatures": seen_signatures,
                    },
                )
                if self.memory is not None:
                    committed_tools = [
                        {"tool": s.action_name, "arguments": s.input_data}
                        for s in self.journal.list_steps(run_id)
                        if s.status == "committed" and s.action_name != "final_answer"
                    ]
                    if committed_tools:
                        self.memory.save_skill(
                            name=self._skill_slug(goal),
                            goal_pattern=goal,
                            steps=committed_tools,
                        )
                return AgentRunResult(
                    run_id=run_id,
                    status=RunStatus.COMPLETED,
                    final_answer=answer,
                    steps_executed=step_idx + 1,
                    history=history,
                )

            tool_name = action.tool_name or ""
            tool = self.tools.get(tool_name)
            if tool is None:
                history.append(
                    {
                        "role": "tool",
                        "content": f"Error: unknown tool {tool_name!r}. Valid tools: {sorted(self.tools.keys())}",
                    }
                )
                step_idx += 1
                continue

            # Loop guard: halt if identical tool+args repeats immediately
            sig = self._call_signature(tool_name, action.arguments)
            if seen_signatures and seen_signatures[-1] == sig:
                self.journal.checkpoint_run(
                    run_id,
                    status=RunStatus.HALTED,
                    current_state=AgentPhase.PLAN,
                    step_index=step_idx,
                    state={
                        "history": history,
                        "context_trust": context_trust.value,
                        "seen_signatures": seen_signatures,
                    },
                )
                return AgentRunResult(
                    run_id=run_id,
                    status=RunStatus.HALTED,
                    final_answer=None,
                    steps_executed=step_idx,
                    halt_reason=f"Loop guard halted repeated call: {sig}",
                    history=history,
                )
            seen_signatures.append(sig)

            # --- AUTHORIZE (AgentSpec + CaMeL taint guard) ---
            wrapped_args = {
                k: (
                    TaintedValue(v, context_trust, "agent_context")
                    if context_trust == TrustLevel.UNTRUSTED_DATA
                    else v
                )
                for k, v in action.arguments.items()
            }
            policy_dec = self.policy.evaluate(
                ActionContext(
                    tool_name=tool_name,
                    capability=tool.capability,
                    arguments=wrapped_args,
                    context_trust=context_trust,
                    workspace_root=self.workspace_root,
                )
            )
            idem_key = f"step-{step_idx}-{tool_name}"
            if policy_dec.action != EnforcementAction.ALLOW:
                self.journal.record_step(
                    run_id,
                    step_index=step_idx,
                    phase=AgentPhase.AUTHORIZE,
                    idempotency_key=idem_key,
                    action_name=tool_name,
                    input_data=action.arguments,
                    output_data=policy_dec.to_dict(),
                    status="blocked",
                )
                self.journal.checkpoint_run(
                    run_id,
                    status=RunStatus.HALTED,
                    current_state=AgentPhase.AUTHORIZE,
                    step_index=step_idx,
                    state={
                        "history": history,
                        "context_trust": context_trust.value,
                        "seen_signatures": seen_signatures,
                    },
                )
                return AgentRunResult(
                    run_id=run_id,
                    status=RunStatus.HALTED,
                    final_answer=None,
                    steps_executed=step_idx,
                    halt_reason=f"Policy {policy_dec.action.value} ({policy_dec.rule_id}): {policy_dec.reason}",
                    history=history,
                )

            # --- EXECUTE (Idempotent replay check) ---
            existing = self.journal.get_step_by_idempotency(run_id, idem_key)
            if existing is not None and existing.status == "committed":
                tool_output = (existing.output_data or {}).get("result")
            else:
                try:
                    clean_args = self.policy.unwrap_arguments(wrapped_args)
                    tool_output = await self._invoke_maybe_async(tool.handler, clean_args)
                    executed_for_saga.append((tool, clean_args, tool_output))
                    consecutive_errors = 0
                except Exception as exc:
                    recoverable = (
                        tool.capability in {Capability.READ, Capability.COMPUTE}
                        and isinstance(
                            exc,
                            (
                                FileNotFoundError,
                                NotADirectoryError,
                                ValueError,
                                TypeError,
                                KeyError,
                            ),
                        )
                        and consecutive_errors < 2
                    )
                    if recoverable:
                        consecutive_errors += 1
                        err_msg = f"ToolError ({type(exc).__name__}): {exc}. Adjust arguments and try again."
                        history.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    action.model_dump(), separators=(",", ":")
                                ),
                            }
                        )
                        history.append(
                            {"role": "tool", "content": f"[{tool_name}] {err_msg}"}
                        )
                        self.journal.record_step(
                            run_id,
                            step_index=step_idx,
                            phase=AgentPhase.EXECUTE,
                            idempotency_key=idem_key,
                            action_name=tool_name,
                            input_data=action.arguments,
                            output_data={"error": err_msg},
                            status="recovered_error",
                        )
                        step_idx += 1
                        self.journal.checkpoint_run(
                            run_id,
                            status=RunStatus.RUNNING,
                            current_state=AgentPhase.PLAN,
                            step_index=step_idx,
                            state={
                                "history": history,
                                "context_trust": context_trust.value,
                                "seen_signatures": seen_signatures,
                            },
                        )
                        continue

                    rolled_back = (
                        await self._rollback_committed(run_id, executed_for_saga)
                        if rollback_on_failure
                        else 0
                    )
                    self.journal.record_step(
                        run_id,
                        step_index=step_idx,
                        phase=AgentPhase.EXECUTE,
                        idempotency_key=idem_key,
                        action_name=tool_name,
                        input_data=action.arguments,
                        output_data={"error": f"{type(exc).__name__}: {exc}"},
                        status="failed",
                    )
                    self.journal.checkpoint_run(
                        run_id,
                        status=RunStatus.FAILED,
                        current_state=AgentPhase.EXECUTE,
                        step_index=step_idx,
                        state={
                            "history": history,
                            "context_trust": context_trust.value,
                            "seen_signatures": seen_signatures,
                        },
                    )
                    return AgentRunResult(
                        run_id=run_id,
                        status=RunStatus.FAILED,
                        final_answer=None,
                        steps_executed=step_idx,
                        halt_reason=f"Tool {tool_name} failed: {exc}",
                        rolled_back_steps=rolled_back,
                        history=history,
                    )

            # --- VERIFY & COMMIT ---
            if tool.returns_untrusted:
                context_trust = TrustLevel.UNTRUSTED_DATA

            obs_text = (
                tool_output
                if isinstance(tool_output, str)
                else json.dumps(tool_output, default=str)
            )
            history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action.model_dump(), separators=(",", ":")),
                }
            )
            history.append({"role": "tool", "content": f"[{tool_name}] {obs_text}"})

            self.journal.record_step(
                run_id,
                step_index=step_idx,
                phase=AgentPhase.COMMIT,
                idempotency_key=idem_key,
                action_name=tool_name,
                input_data=action.arguments,
                output_data={"result": tool_output},
                status="committed",
            )
            step_idx += 1
            self.journal.checkpoint_run(
                run_id,
                status=RunStatus.RUNNING,
                current_state=AgentPhase.PLAN,
                step_index=step_idx,
                state={
                    "history": history,
                    "context_trust": context_trust.value,
                    "seen_signatures": seen_signatures,
                },
            )

        self.journal.checkpoint_run(
            run_id,
            status=RunStatus.HALTED,
            current_state=AgentPhase.PLAN,
            step_index=step_idx,
            state={
                "history": history,
                "context_trust": context_trust.value,
                "seen_signatures": seen_signatures,
            },
        )
        return AgentRunResult(
            run_id=run_id,
            status=RunStatus.HALTED,
            final_answer=None,
            steps_executed=step_idx,
            halt_reason=f"Reached max_steps={self.max_steps}",
            history=history,
        )
