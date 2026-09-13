# mmWave → Cardiopulmonary ingest audit v1

Status: **code/contract audit implemented; governed-cohort real rerun pending local snapshot access**.

Authority:

- `FocusWave-Formal-Analysis@codex/code-fix-ledger/分析设计/1.16.10-监督学习模态与设备定义修订及代码迁移计划_20260912.md`
- `FocusWave-Formal-Analysis@codex/code-fix-ledger/分析设计/1.16.11-监督学习特征文件接口与运行前闸门_20260913.md`
- producer `greenboo26/focuswave-multimodal-attention-analysis@main`
- producer artifact `MMWAVE_INTEGRATION_SNAPSHOT_V1`
- implementation issue: `https://github.com/kyandi233-dev/Attention-Analysis/issues/73`

## 1. Scope

This bridge answers an engineering question only: can the current versioned mmWave snapshot enter the formal FocusWave analysis stack without changing participant identity, probe keys, temporal meaning, or producer availability/error semantics?

It does **not** re-run the radar estimator, optimize HR, create snapshot v2, qualify HRV, run LOSO, or promote HR/BR from the current physiology-limited/supporting-only role.

The semantic contract is:

```text
scientific modality = cardiopulmonary
source_namespace = mmwave
required_devices = [mmwave]
```

The current formal science quantities are:

- `mmwave_hr_fused_bpm_median` — radar-derived fused heart-rate estimate;
- `mmwave_breath_rate_breaths_per_min_median` — radar-derived respiration-rate estimate.

`mmwave_motion_proxy_median` remains diagnostic-only and is not a Movement predictor. HRV quantities remain blocked.

## 2. Governed denominator and producer states

The canonical snapshot contract is fixed at:

- 116 governed sessions;
- 61 participant groups;
- 2,320 governed probes;
- 2,180 AVAILABLE probes from 109 sessions;
- 40 SOURCE_UNAVAILABLE probes from 2 sessions;
- 100 SOURCE_MALFORMED probes from 5 sessions.

All 2,320 governed probe keys must remain present. The 40 unavailable rows and 100 malformed rows both retain missing HR/BR values, but they are **not collapsed into one missingness class**:

- `SOURCE_UNAVAILABLE`: the source is absent and enters the current quality interface as `structural_source_missing`;
- `SOURCE_MALFORMED`: a source exists but cannot be read/used and enters as `structural_source_unreadable`.

This distinction follows the producer replacement contract, which prohibits silently reclassifying error rows as ordinary source absence. Numeric zero-fill, stale carry-forward, silent row deletion, or ordinary median imputation fails the ingest contract.

Participant identity comes from the snapshot's canonical `participant_group_id` and is checked against the Behavior-authoritative probe table. The adapter never reconstructs participant identity from folder/session naming.

## 3. Time-legality contract

Producer-side science alignment uses the DLL host receive/enqueue Unix clock. The Python processing timestamp is QC-only.

For every snapshot row the adapter requires:

```text
window_name == pre_30s
alignment_clock_source == dll_host_receive_enqueue
window_nominal_start_unix_ms == probe_onset_unix_ms - 30000 ms
window_effective_start_unix_ms >= window_nominal_start_unix_ms
window_effective_start_unix_ms < probe_onset_unix_ms
window_end_unix_ms == probe_onset_unix_ms
```

The producer field-role/replacement contract defines `window_end_unix_ms` as the right-exclusive probe onset, so the scientific interval is:

```text
[window_effective_start_unix_ms, probe_onset_unix_ms)
```

`window_effective_start_unix_ms > window_nominal_start_unix_ms` is legal when the nominal 30 s interval is truncated at formal Block start. Passing the row checks plus the producer contract yields `verified_pre_probe_only` provenance for the snapshot representation. This time qualification does not grant supervised-prediction eligibility; physiological qualification is a separate gate.

## 4. Reused current interfaces

The implementation reuses current 1.16 code rather than replacing it:

1. `quality_admission.audit_quality()` for source/readability/QC/computability states;
2. `analysis_sets.build_analysis_sets()` with explicit `required_feature_records` that map scientific modality `cardiopulmonary` to source namespace `mmwave`;
3. comparison-specific membership from the exact HR/BR predictor union.

The interface smoke consumes no Q1/Q2 and trains no model. It checks only engineering membership semantics. Unrelated Ocular or Movement availability is never referenced.

## 5. Feature-handoff boundary

`mmwave_cardiopulmonary_feature_handoff.csv` exposes HR/BR in the final semantic shape but is **not** the final unified feature registry. It records scientific modality, source namespace, required device, temporal provenance, physiology qualification, registry readiness, and researcher-freeze requirement.

All prediction eligibility flags remain `false`. Current physiology qualification is `LIMITED_SUPPORTING_ONLY`.

## 6. Outputs

```text
mmwave_cardiopulmonary_ingest_audit.csv
mmwave_cardiopulmonary_coverage.csv
mmwave_cardiopulmonary_feature_handoff.csv
mmwave_cardiopulmonary_taskb_source.csv
mmwave_cardiopulmonary_ingest_manifest.json
interface_smoke/formal_probe_identity.csv
interface_smoke/modality_probe_status.csv
interface_smoke/probe_feature_status.csv
interface_smoke/feature_coverage.csv
interface_smoke/modality_availability.csv
interface_smoke/analysis_sets.csv
interface_smoke/analysis_set_summary.csv
```

The manifest reports the three producer states separately. The neutral aggregate `retained_missing_or_error_probe_n` counts rows that retain missing HR/BR without erasing whether the reason was source absence or malformed source error.

## 7. Governed-cohort local run

The 2,320-row snapshot is producer-recorded local-only and is not committed to GitHub or the accessible Drive snapshot folder. Therefore CI and cloud review cannot claim a fresh governed-cohort execution of this adapter.

On the machine holding the current snapshot:

```powershell
conda activate attention-behavior-formal
cd "D:\Project\厚粲杯\08_算法\Attention-Analysis"

git fetch origin
git switch codex/mmwave-cardiopulmonary-ingest-audit-v1
git pull

python scripts/mmwave_cardiopulmonary_ingest_audit.py `
  --snapshot "D:\Project\厚粲杯\11_数据\_FormalAnalysis\mmWave\mmwave_integration_snapshot_v1_20260912_r4\mmwave_probe_merge_ready.csv" `
  --behavior-probes "<CURRENT_BEHAVIOR_116_SESSION_PROBE_TABLE>" `
  --output-root "D:\Project\厚粲杯\11_数据\_FormalAnalysis\mmWave\mmwave_cardiopulmonary_ingest_audit_v1"
```

Do not use `--allow-subset-smoke` for the governed-cohort gate. A strict run fails unless it conserves exactly 2,320 keys / 116 sessions / 61 groups, reproduces 2,180 AVAILABLE + 40 SOURCE_UNAVAILABLE + 100 SOURCE_MALFORMED rows, keeps HR and BR finite exactly 2,180 times each, and yields exactly 2,180 probes in both complete and missing-aware Cardiopulmonary interface sets.

`--allow-subset-smoke` is only for synthetic/schema smoke. Such a run writes `engineering_integration_qualification=PENDING_GOVERNED_REAL_RUN`; it cannot be used as release evidence.

## 8. Release interpretation

Only a successful strict governed-cohort rerun supports:

```text
engineering integration qualification = READY
physiology qualification = LIMITED / SUPPORTING_ONLY
```

It still does not authorize Cardiopulmonary predictors in the final supervised model. That later decision remains a Formal-method/researcher freeze after the real audit artifacts are reviewed. Interface readiness and physiological measurement validity remain separate states.
