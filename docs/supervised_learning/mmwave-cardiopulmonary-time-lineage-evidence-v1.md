# mmWave Cardiopulmonary time-lineage evidence v1

Status: `blocked_upstream_contract_mismatch` for formal prediction; snapshot v1 remains usable for interface/schema/supporting integration.

This note separates three different claims that must not be conflated:

1. the snapshot's declared metadata timing is internally consistent;
2. producer-side corrected-replay evidence indicates the intended right-open pre-probe window was used;
3. the exact executed source lineage is still not reproducible without contradiction from the current remote repository.

## 1. Current producer main and snapshot execution lineage differ

The currently visible producer `main` adapter still reflects the older implementation line. Formal `1.15.9` records the unresolved contract items: use of the wrong visible slicing semantics in current main, incomplete current-main source closure, and a usable-window field whose semantics are not yet canonical.

`MMWAVE_INTEGRATION_SNAPSHOT_V1`, however, was packaged from a corrected DLL-time replay rather than from that visible legacy current-main adapter.

## 2. Existing corrected-replay evidence

Durable producer evidence records the intended execution semantics as:

```text
science clock = DLL host receive/enqueue timestamp
Python processing timestamp = QC only
left boundary = searchsorted(..., side='left')
right boundary = searchsorted(..., side='left')
window = [window_effective_start_unix_ms, probe_onset_unix_ms)
formal Block start = front-boundary clipping; no cross-Block window
```

The producer P0 audit also records that, on the replay comparison, probes with identical frame membership retained identical deterministic derived values. This is evidence in favor of the intended pre-probe semantics.

## 3. Downstream metadata endpoint audit

PR #76 performed a strict governed-cohort downstream audit of the declared endpoint fields:

```text
2320 probes / 116 sessions / 61 participant groups
window_end_unix_ms - probe_onset_unix_ms:
  n_zero = 2320
  n_nonzero = 0
  min = max = median = 0 ms
```

The downstream metadata contract is therefore frozen as exact identity:

```text
window_end_unix_ms == probe_onset_unix_ms
```

This closes the previously defective NumPy relative-tolerance guard. It does **not** prove which individual frames actually entered the upstream estimator.

## 4. Source-code provenance remains unresolved

Formal `1.15.9` records a stricter second-round finding: the replay manifest's recorded source hashes do not match the files obtained from its self-declared `source_commit`, and the exact historical execution commit is not currently recoverable from the remote lineage without ambiguity.

Therefore the current formal interpretation is:

```text
metadata contract = verified
engineering integration = READY after strict real run
formal prediction time-legality = blocked_upstream_contract_mismatch
source-code provenance closure = pending
frame-membership proof = pending
```

This is not evidence that snapshot v1 used future information. It is evidence that the executed producer source cannot yet be reproduced and tied to the formal contract with the level of provenance required for prediction eligibility.

## 5. Required second-track producer evidence

Promotion to `verified_pre_probe_only` requires an upstream artifact keyed by the canonical probe identity that can establish actual selected-frame membership, not just declared metadata. At minimum it should carry:

```text
canonical probe key
selected_frame_count
first_selected_timestamp
last_selected_timestamp
all_selected_timestamp_lt_probe_onset
frame_membership_digest
producer commit
source file hashes / run id
```

The producer repair must also provide an old-vs-new per-probe audit so that changed and unchanged frame memberships, HR/BR/QC transitions, and missing/error states can be traced explicitly.

## 6. Downstream rule

Until that producer/provenance work is accepted:

- `MMWAVE_INTEGRATION_SNAPSHOT_V1` may be used for interface, schema, denominator, missingness, and supporting integration work;
- HR/BR remain `LIMITED_SUPPORTING_ONLY`;
- `time_legality_status = blocked_upstream_contract_mismatch`;
- all formal prediction-eligibility flags remain false;
- HRV remains blocked;
- mmWave motion remains diagnostic-only and is not a qualified Movement feature.

No downstream metadata tolerance, including an endpoint tolerance, may substitute for actual producer frame-membership evidence.
