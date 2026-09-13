# mmWave → Cardiopulmonary ingest audit v1

Status: **engineering interface implemented; metadata endpoint contract frozen; formal prediction time-legality blocked upstream**.

Authority:

- `FocusWave-Formal-Analysis@codex/code-fix-ledger/分析设计/1.15.9-毫米波生成链合同修复与time-legality处置_20260913.md`
- `FocusWave-Formal-Analysis@codex/code-fix-ledger/分析设计/1.16.10-监督学习模态与设备定义修订及代码迁移计划_20260912.md`
- `FocusWave-Formal-Analysis@codex/code-fix-ledger/分析设计/1.16.11-监督学习特征文件接口与运行前闸门_20260913.md`
- producer `greenboo26/focuswave-multimodal-attention-analysis@main`
- artifact `MMWAVE_INTEGRATION_SNAPSHOT_V1`

## 1. Scope

This bridge answers only the engineering-ingest question: can the current versioned mmWave snapshot enter the FocusWave common interface without changing participant identity, canonical probe keys, producer availability/error semantics, or declared metadata timing?

It does not re-run the radar estimator, optimize HR/BR, qualify HRV, train LOSO models, promote mmWave motion to Movement, or mutate the final unified feature registry.

```text
scientific_modality = cardiopulmonary
source_namespace = mmwave
required_devices = [mmwave]
physiology_qualification = LIMITED_SUPPORTING_ONLY
```

## 2. Governed denominator and producer states

The strict canonical gate remains:

- 116 sessions;
- 61 participant groups;
- 2,320 probes;
- 2,180 `AVAILABLE`;
- 40 `SOURCE_UNAVAILABLE`;
- 100 `SOURCE_MALFORMED`.

All 2,320 canonical probe keys are retained. `SOURCE_UNAVAILABLE` maps to `structural_source_missing`; `SOURCE_MALFORMED` maps to `structural_source_unreadable`. Both preserve HR/BR as NaN. Zero fill, stale carry-forward, silent deletion, and ordinary imputation are forbidden.

Participant identity is checked against the Behavior-authoritative table; it is never inferred from folder or session names.

## 3. Metadata timing contract

The governed-cohort endpoint audit run at PR #76 commit `53f642ab7c76e6692a11a68636bf36f1158418d5` found:

```text
n_total = 2320
n_zero = 2320
n_nonzero = 0
min = max = median = 0 ms
```

Therefore the downstream metadata contract is now frozen as exact integer-millisecond identity:

```text
window_end_unix_ms == probe_onset_unix_ms
```

No positive or negative non-zero endpoint delta is admitted. The previous temporary 1 ms absolute tolerance is retired. This exact endpoint check is a metadata identity check only.

The row audit additionally requires:

```text
window_name == pre_30s
alignment_clock_source == dll_host_receive_enqueue
probe_onset_unix_ms - window_nominal_start_unix_ms == 30000
window_effective_start_unix_ms >= window_nominal_start_unix_ms
window_effective_start_unix_ms < probe_onset_unix_ms
```

## 4. Formal time-legality remains blocked

Passing the metadata checks does **not** establish `verified_pre_probe_only`.

Per Formal `1.15.9`, the current Cardiopulmonary feature status is:

```text
engineering_integration_qualification = READY   # after strict governed run
time_legality_status = blocked_upstream_contract_mismatch
physiology_qualification = LIMITED_SUPPORTING_ONLY
registry_ready = false
prediction eligibility = false
```

The blocker is upstream reproducibility, not an observed metadata endpoint mismatch. Before promotion to `verified_pre_probe_only`, the producer side must close all of the following:

1. current-main contract repair;
2. strict right-open frame selection `[window_effective_start, probe_onset)`;
3. old-vs-new per-probe frame-membership audit;
4. actual selected-frame evidence proving no `timestamp >= probe_onset`;
5. source-code provenance closure linking the executed code, commit, hashes, and run manifest without contradiction.

The current downstream snapshot contains no per-frame timestamps or frame-membership digest, so this adapter cannot perform that second-track proof itself.

## 5. Behavior probe-index compatibility

Behavior science-v3 uses `probe_order_in_block`; the common interface uses `probe_index_in_block`. The CLI applies an explicit alias only when the canonical column is absent.

If both columns are present, they must agree row-for-row. Any disagreement fails closed rather than silently preferring one column.

## 6. Outputs

```text
mmwave_cardiopulmonary_ingest_audit.csv
mmwave_cardiopulmonary_coverage.csv
mmwave_cardiopulmonary_feature_handoff.csv
mmwave_cardiopulmonary_taskb_source.csv
mmwave_cardiopulmonary_ingest_manifest.json
endpoint_delta_ms.csv
endpoint_delta_nonzero_ms.csv
endpoint_delta_summary.json
interface_smoke/*
```

## 7. Strict local rerun

The row-level snapshot is local-only. After the exact-endpoint/time-status correction commit, rerun the strict governed audit on the machine holding the data:

```powershell
cd "D:\Project\厚粲杯\08_算法\Attention-Analysis"
git fetch origin
git switch codex/mmwave-cardiopulmonary-endpoint-guard-v1
git pull

python scripts/mmwave_cardiopulmonary_ingest_audit.py `
  --snapshot "D:\Project\厚粲杯\11_数据\_FormalAnalysis\mmWave\mmwave_integration_snapshot_v1_20260912_r4\MMWAVE_INTEGRATION_SNAPSHOT_V1_PROBES_LOCAL_ONLY.csv" `
  --behavior-probes "D:\Project\厚粲杯\11_数据\_FormalAnalysis\Behavior\formal_v3\probe_primary_30s.csv" `
  --output-root "D:\Project\厚粲杯\11_数据\_FormalAnalysis\mmWave\mmwave_cardiopulmonary_endpoint_guard_v1_20260913"
```

Do not use `--allow-subset-smoke` for the governed-cohort gate. The refreshed run is expected to preserve 2,320 / 116 / 61, 2,180 / 40 / 100, and exact endpoint delta 0 on all 2,320 probes. Any deviation must be investigated rather than coerced to the historical counts.

## 8. Release interpretation

This PR can close the downstream engineering interface and metadata endpoint guard. It cannot close the producer provenance/time-lineage task and cannot authorize Cardiopulmonary predictors in formal supervised learning.
