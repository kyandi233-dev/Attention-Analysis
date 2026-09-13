# mmWave Cardiopulmonary time-lineage evidence v1

Status: `verified_pre_probe_only` for `MMWAVE_INTEGRATION_SNAPSHOT_V1`, with a separate source-reproducibility limitation.

This note exists because two superficially conflicting code states must not be conflated.

## 1. The visible current-main legacy adapter is not the snapshot-v1 execution source

The file currently visible on producer `main` at:

`greenboo26/focuswave-multimodal-attention-analysis/scripts/maintenance/run_mmwave_probe_merge_ready_20260831.py`

still contains the older semantics:

- reads `window_start_unix_ms` rather than using the corrected effective start for slicing;
- slices against timestamp column index 2 (Python worker processing time);
- uses `searchsorted(..., side="right")` at the right boundary.

Those statements are true of the visible legacy file, but they must **not** be projected onto `MMWAVE_INTEGRATION_SNAPSHOT_V1`.

## 2. Snapshot v1 is explicitly packaged from a corrected local DLL-time replay

The tracked snapshot manifest records:

```text
source_commit = 16729b2ef245f9304dae8674f3bac433bc02e98c
reuse_gate = PASS: reused corrected DLL-time replay ...; no algorithm rerun
input = mmwave_b1_formal_dll_replay_20260912_r2
```

The current tracked snapshot builder also hard-codes the same producer commit and packages the immutable J72 + E44 corrected replay tables. It validates the exported row fields:

```text
window_end_unix_ms == probe_onset_unix_ms
window_nominal_start_unix_ms + 30000 == window_end_unix_ms
window_effective_start_unix_ms >= window_nominal_start_unix_ms
window_effective_start_unix_ms < window_end_unix_ms
alignment_clock_source = dll_host_receive_enqueue
```

The builder labels the adapter version `issue34-dll-cutover-v1`; it does not re-estimate the radar signals.

## 3. The corrected execution semantics were audited separately at producer P0

The durable producer evidence:

- `docs/canonical/2026-09-12_MMWAVE_HR_RECOVERY_P0_PIPELINE_LINEAGE_AUDIT.md`
- `docs/canonical/2026-09-12_MMWAVE_HR_RECOVERY_P0_MANIFEST.json`

records the exact current-formal source as `16729b2ef245f9304dae8674f3bac433bc02e98c` and states that the B1 DLL replay used:

```text
scientific timestamp = DLL host receive/enqueue column 1
Python worker timestamp = QC only
left boundary = searchsorted(..., side='left')
right boundary = searchsorted(..., side='left')
window = [probe_end - 30 s, probe_end)
formal Block start = front-boundary clipping; no cross-Block window
```

For B1 the audit records:

```text
116 sessions
2320 rows
2180 computable probes
changed frame membership = 1847
identical frame membership = 333
deterministic feature changed when membership identical = 0
regression gate = PASS
models trained = false
```

This is the producer-side evidence required by the Formal rule that `MMWAVE_INTEGRATION=READY` alone cannot establish time legality.

## 4. Source reproducibility limitation

The same P0 manifest records:

```text
16729b2... reachable_from_origin_main_at_audit_start = false
```

and the current GitHub remote still cannot resolve that exact commit. Therefore the correct downstream statement is two-part:

```text
time-legality status = verified_pre_probe_only
source reproducibility status = durable producer audit present / exact execution commit not remote-reachable
```

The second clause is a provenance/reproducibility limitation. It is not evidence that snapshot v1 used future information, because the exact corrected replay was separately content-addressed and audited before snapshot packaging. Conversely, current `main` must not be presented as the executable source of snapshot v1 unless a semantics-equivalent corrected commit is later restored to the remote repository and verified.

## 5. Downstream rule for this PR

The mmWave → Cardiopulmonary ingest audit therefore:

1. verifies every snapshot row still reports the frozen `pre_30s` window fields and DLL clock;
2. carries `verified_pre_probe_only` only for the versioned `MMWAVE_INTEGRATION_SNAPSHOT_V1` lineage;
3. preserves the producer commit/run provenance in the manifest;
4. keeps all HR/BR prediction-eligibility flags false because physiological qualification is a separate gate;
5. must not infer that a future snapshot or a newly generated table is time-legal merely because it is called mmWave or because the visible current-main legacy adapter exists.

A future replacement snapshot must pass its own producer time-lineage audit under the replacement contract; this v1 evidence is not transferable by feature name alone.
