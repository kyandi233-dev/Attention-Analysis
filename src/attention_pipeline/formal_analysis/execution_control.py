from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, TypeVar

import pandas as pd

T = TypeVar("T")


@dataclass
class StepRecord:
    step: str
    required: bool
    requested: bool
    status: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    elapsed_seconds: float | None = None
    skip_reason: str = ""
    error_type: str = ""
    error_message: str = ""


class ExecutionLedger:
    """Record formal-pipeline step timing and explicit skip/failure state.

    A skipped optional analysis is not a failure, but it makes the run partial.
    Required structural steps are never silently skipped.  Exceptions are
    recorded before being re-raised so local execution can still diagnose the
    exact failing stage.
    """

    def __init__(self, *, pipeline: str) -> None:
        self.pipeline = pipeline
        self.records: list[StepRecord] = []

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def run(
        self,
        step: str,
        func: Callable[[], T],
        *,
        required: bool,
        requested: bool = True,
        skip_reason: str = "not requested",
    ) -> T | None:
        if not requested:
            if required:
                raise ValueError(f"required formal step cannot be skipped: {step}")
            self.records.append(
                StepRecord(
                    step=step,
                    required=False,
                    requested=False,
                    status="skipped",
                    skip_reason=skip_reason,
                )
            )
            return None

        started_at = self._now()
        started = time.perf_counter()
        try:
            value = func()
        except Exception as exc:
            finished = time.perf_counter()
            self.records.append(
                StepRecord(
                    step=step,
                    required=required,
                    requested=True,
                    status="failed",
                    started_at_utc=started_at,
                    finished_at_utc=self._now(),
                    elapsed_seconds=float(finished - started),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            raise
        finished = time.perf_counter()
        self.records.append(
            StepRecord(
                step=step,
                required=required,
                requested=True,
                status="completed",
                started_at_utc=started_at,
                finished_at_utc=self._now(),
                elapsed_seconds=float(finished - started),
            )
        )
        return value

    @property
    def run_status(self) -> str:
        if any(row.status == "failed" for row in self.records):
            return "failed"
        if any(row.status == "skipped" for row in self.records):
            return "partial_run"
        return "complete"

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self.records])

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "run_status": self.run_status,
            "steps": [asdict(row) for row in self.records],
        }

    def write(self, output_root: Path) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        self.as_frame().to_csv(
            output_root / "execution_steps.csv", index=False, encoding="utf-8-sig"
        )
        (output_root / "execution_manifest.json").write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def parse_step_names(raw: str | None) -> set[str]:
    if raw is None or not str(raw).strip():
        return set()
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def resolve_optional_steps(
    available: Iterable[str],
    *,
    only: str | None = None,
    skip: str | None = None,
) -> set[str]:
    """Resolve optional analyses requested by ``--only-steps``/``--skip-steps``.

    With no arguments all optional steps run.  ``only`` narrows the optional
    set.  ``skip`` is then removed from that set.  Unknown names fail closed.
    Required structural steps are intentionally managed by the caller and are
    not part of this selector.
    """
    available_set = set(available)
    only_set = parse_step_names(only)
    skip_set = parse_step_names(skip)
    unknown = (only_set | skip_set) - available_set
    if unknown:
        raise ValueError(
            "unknown optional formal step(s): " + ", ".join(sorted(unknown))
        )
    selected = set(available_set) if not only_set else set(only_set)
    return selected - skip_set
