"""Sample App 3: Memory Companion — CoALA bi-temporal recall & sleep-time consolidation."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from strata.config import StrataConfig
from strata.sdk import StrataClient


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="strata_companion_") as tmp:
        cfg = StrataConfig(data_dir=Path(tmp))
        async with StrataClient.local(cfg, prefer_onnx=False) as client:
            await client.remember("demo", "user", "My name is Dev. I live in Pune.")
            await client.remember("demo", "user", "I prefer local-first Python tools.")
            await client.remember("demo", "user", "I moved to Bengaluru.")

            consolidation = await client.consolidate()
            print("Consolidation:", json.dumps(consolidation, indent=2))

            facts = await client.recall("Where does Dev live?")
            print("Recalled facts:", json.dumps(facts, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
