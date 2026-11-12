"""Sample App 2: Ops Helper — 8 GB health inspection & ResourceGovernor telemetry."""

from __future__ import annotations

import asyncio
import json

from strata.config import StrataConfig
from strata.perf.bench import run_doctor


async def main() -> None:
    cfg = StrataConfig.from_env()
    report = await run_doctor(cfg)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
