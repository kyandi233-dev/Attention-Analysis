from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


_INT_FIELDS = {
    "block_num",
    "trial_num",
    "cycle_num",
    "position_in_cycle",
    "is_no_go",
    "response",
    "correct",
    "commission",
    "omission",
    "is_probe",
    "probe_response",
    "probe_vigilance",
    "absolute_onset_time",
    "response_time",
    "probe_onset_time",
    "probe_response_time",
}
_FLOAT_FIELDS = {
    "stimulus_size",
    "rt",
    "probe_rt",
    "probe_vigilance_rt",
    "block_onset_time",
    "rest_duration",
}
_CONTEXT_FIELDS = [
    "trial_num",
    "condition",
    "cycle_num",
    "position_in_cycle",
    "stimulus_name",
    "stimulus_size",
    "is_no_go",
    "response",
    "correct",
    "commission",
    "omission",
    "is_probe",
    "probe_response",
    "probe_vigilance",
]


def _parse_int(value: object) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_behavior_trials(path: Path) -> list[dict[str, object]]:
    """Read one FocusWave formal behavior CSV without discarding available fields."""
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in _INT_FIELDS:
                    parsed[key] = _parse_int(value)
                elif key in _FLOAT_FIELDS:
                    parsed[key] = _parse_float(value)
                else:
                    parsed[key] = str(value or "")
            onset = parsed.get("absolute_onset_time")
            if isinstance(onset, int):
                records.append(parsed)
    records.sort(key=lambda item: int(item["absolute_onset_time"]))
    return records


@dataclass(frozen=True)
class BehaviorIndex:
    records: tuple[dict[str, object], ...]
    onsets: tuple[int, ...]

    @classmethod
    def from_csv(cls, path: Path | None) -> "BehaviorIndex | None":
        if path is None or not path.exists():
            return None
        records = read_behavior_trials(path)
        if not records:
            return None
        return cls(
            records=tuple(records),
            onsets=tuple(int(record["absolute_onset_time"]) for record in records),
        )

    def context_at(self, unix_ms: int, *, trial_duration_ms: int = 1150) -> dict[str, object]:
        """Map a frame to the latest trial and explicit trial/probe temporal state.

        The latest trial metadata remains available between trials so later analyses
        can reconstruct trial-centred windows. ``trial_active`` and ``probe_active``
        state whether the current frame is actually inside the nominal trial or the
        recorded probe display interval; this avoids silently labelling probe/recovery
        time as stimulus time.
        """
        idx = bisect_right(self.onsets, unix_ms) - 1
        if idx < 0:
            return empty_behavior_context()

        record = self.records[idx]
        trial_onset = self.onsets[idx]
        next_onset = self.onsets[idx + 1] if idx + 1 < len(self.onsets) else None
        time_from_trial = unix_ms - trial_onset
        time_to_next = next_onset - unix_ms if next_onset is not None else None
        trial_active = 0 <= time_from_trial < int(trial_duration_ms)

        probe_onset = record.get("probe_onset_time")
        probe_response = record.get("probe_response_time")
        probe_active = bool(
            isinstance(probe_onset, int)
            and isinstance(probe_response, int)
            and probe_onset <= unix_ms <= probe_response
        )

        if probe_active:
            behavior_state = "probe"
        elif trial_active:
            behavior_state = "trial"
        elif (
            isinstance(probe_response, int)
            and unix_ms > probe_response
            and (next_onset is None or unix_ms < next_onset)
        ):
            behavior_state = "post_probe_recovery"
        else:
            behavior_state = "intertrial"

        context = empty_behavior_context()
        for field in _CONTEXT_FIELDS:
            context[field] = record.get(field)
        context.update(
            {
                "trial_onset_unix_ms": trial_onset,
                "time_from_trial_onset_ms": time_from_trial,
                "next_trial_onset_unix_ms": next_onset,
                "time_to_next_trial_onset_ms": time_to_next,
                "trial_active": trial_active,
                "probe_onset_unix_ms": probe_onset if isinstance(probe_onset, int) else None,
                "probe_response_unix_ms": probe_response if isinstance(probe_response, int) else None,
                "probe_active": probe_active,
                "behavior_state": behavior_state,
            }
        )
        return context


def empty_behavior_context() -> dict[str, object]:
    context: dict[str, object] = {field: None for field in _CONTEXT_FIELDS}
    context["condition"] = None
    context["stimulus_name"] = None
    context.update(
        {
            "trial_onset_unix_ms": None,
            "time_from_trial_onset_ms": None,
            "next_trial_onset_unix_ms": None,
            "time_to_next_trial_onset_ms": None,
            "trial_active": False,
            "probe_onset_unix_ms": None,
            "probe_response_unix_ms": None,
            "probe_active": False,
            "behavior_state": None,
        }
    )
    return context
