"""Durable SQLite step journal with idempotency keys, resume, and Saga rollbacks."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable

from strata.storage.db import StrataDB


class AgentPhase(StrEnum):
    PLAN = "PLAN"
    VALIDATE = "VALIDATE"
    AUTHORIZE = "AUTHORIZE"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    HALTED = "halted"


@dataclass(slots=True)
class RunRecord:
    run_id: str
    goal: str
    status: RunStatus
    current_state: AgentPhase
    step_index: int
    state: dict[str, Any]
    created_at: float
    updated_at: float


@dataclass(slots=True)
class StepRecord:
    step_index: int
    phase: AgentPhase
    idempotency_key: str
    action_name: str | None
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    status: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d


class DurableStepJournal:
    """Crash-resistant SQLite execution journal supporting idempotent replay & Saga rollback."""

    def __init__(self, db: StrataDB) -> None:
        self._db = db

    def create_run(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> RunRecord:
        rid = run_id or uuid.uuid4().hex[:12]
        now = time.time()
        state = initial_state or {}
        with self._db.conn:
            self._db.execute(
                """
                INSERT INTO agent_runs (
                    run_id, goal, status, current_state, step_index, state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    rid,
                    goal,
                    RunStatus.RUNNING.value,
                    AgentPhase.PLAN.value,
                    json.dumps(state),
                    now,
                    now,
                ),
            )
        return RunRecord(
            run_id=rid,
            goal=goal,
            status=RunStatus.RUNNING,
            current_state=AgentPhase.PLAN,
            step_index=0,
            state=state,
            created_at=now,
            updated_at=now,
        )

    def load_run(self, run_id: str) -> RunRecord | None:
        row = self._db.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            goal=row["goal"],
            status=RunStatus(row["status"]),
            current_state=AgentPhase(row["current_state"]),
            step_index=int(row["step_index"]),
            state=json.loads(row["state_json"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def checkpoint_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        current_state: AgentPhase,
        step_index: int,
        state: dict[str, Any],
    ) -> None:
        now = time.time()
        with self._db.conn:
            self._db.execute(
                """
                UPDATE agent_runs
                SET status = ?, current_state = ?, step_index = ?, state_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    status.value,
                    current_state.value,
                    step_index,
                    json.dumps(state),
                    now,
                    run_id,
                ),
            )

    def get_step_by_idempotency(self, run_id: str, idempotency_key: str) -> StepRecord | None:
        row = self._db.execute(
            "SELECT * FROM agent_steps WHERE run_id = ? AND idempotency_key = ?",
            (run_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        return StepRecord(
            step_index=int(row["step_index"]),
            phase=AgentPhase(row["phase"]),
            idempotency_key=row["idempotency_key"],
            action_name=row["action_name"],
            input_data=json.loads(row["input_json"]),
            output_data=json.loads(row["output_json"]) if row["output_json"] else None,
            status=row["status"],
            created_at=float(row["created_at"]),
        )

    def record_step(
        self,
        run_id: str,
        *,
        step_index: int,
        phase: AgentPhase,
        idempotency_key: str,
        action_name: str | None,
        input_data: dict[str, Any],
        output_data: dict[str, Any] | None,
        status: str,
    ) -> None:
        now = time.time()
        with self._db.conn:
            self._db.execute(
                """
                INSERT INTO agent_steps (
                    run_id, step_index, phase, idempotency_key, action_name,
                    input_json, output_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, idempotency_key) DO UPDATE SET
                    phase = excluded.phase,
                    output_json = excluded.output_json,
                    status = excluded.status
                """,
                (
                    run_id,
                    step_index,
                    phase.value,
                    idempotency_key,
                    action_name,
                    json.dumps(input_data),
                    json.dumps(output_data) if output_data is not None else None,
                    status,
                    now,
                ),
            )

    def list_steps(self, run_id: str) -> list[StepRecord]:
        rows = self._db.execute(
            "SELECT * FROM agent_steps WHERE run_id = ? ORDER BY step_index ASC, id ASC",
            (run_id,),
        ).fetchall()
        return [
            StepRecord(
                step_index=int(r["step_index"]),
                phase=AgentPhase(r["phase"]),
                idempotency_key=r["idempotency_key"],
                action_name=r["action_name"],
                input_data=json.loads(r["input_json"]),
                output_data=json.loads(r["output_json"]) if r["output_json"] else None,
                status=r["status"],
                created_at=float(r["created_at"]),
            )
            for r in rows
        ]

    def mark_steps_rolled_back(self, run_id: str) -> int:
        with self._db.conn:
            cur = self._db.execute(
                "UPDATE agent_steps SET status = 'rolled_back' WHERE run_id = ? AND status = 'committed'",
                (run_id,),
            )
            return cur.rowcount


SagaCompensator = Callable[[StepRecord], Awaitable[None]]
