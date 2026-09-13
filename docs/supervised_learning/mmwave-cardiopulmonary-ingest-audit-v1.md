# mmWave → Cardiopulmonary ingest audit v1

Status: **code/contract audit implemented; governed-cohort real rerun pending local snapshot access**.

Authority:

- `FocusWave-Formal-Analysis@codex/code-fix-ledger/分析设计/1.16.10-监督学习模态与设备定义修订及代码迁移计划_20260912.md`
- `FocusWave-Formal-Analysis@codex/code-fix-ledger/分析设计/1.16.11-监督学习特征文件接口与运行前闸门_20260913.md`
- producer `greenboo26/focuswave-multimodal-attention-analysis@main`
- producer artifact `MMWAVE_INTEGRATION_SNAPSHOT_V1`
- implementation issue: `https://github.com/kyandi233-dev/Attention-Analysis/issues/73`

## 1. Scope

This bridge answers an engineering question only: can the current versioned mmWave snapshot enter the formal FocusWave analysis stack without changing participant identity, probe keys, temporal meaning, or structural missingness?

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

## 2. Governed denominator and missingness

The canonical snapshot contract is fixed at:

- 116 governed sessions;
- 61 participant groups;
- 2,320 governed probes;
- 2,180 AVAILABLE probes from 109 sessions;
- 40 SOURCE_UNAVAILABLE probes from 2 sessions;
- 100 SOURCE_MALFORMED probes from 5 sessions.

The adapter requires all governed probe keys to remain present. `SOURCE_UNAVAILABLE` and `SOURCE_MALFORMED` are structural measurement states: their HR/BR values must remain missing. Numeric zero-fill or stale numeric carry-forward fails the ingest audit.

Participant identity is read from the snapshot's canonical `participant_group_id` and checked against the Behavior-authoritative probe table. The adapter never reconstructs participant identity from folder/session naming.

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

The producer's field-role/replacement contract defines `window_end_unix_ms` as the **right-exclusive** probe onset, so the scientific interval is:

```text
[window_effective_start_unix_ms, probe_onset_unix_ms)
```

`window_effective_start_unix_ms > window_nominal_start_unix_ms` is legal when the nominal 30 s window is truncated at the formal Block start. The downstream generic audit is deliberately given `window_seconds_nominal=30` rather than falsely asserting that every effective interval is exactly 30 s.

Passing this audit yields time-provenance status `verified_pre_probe_only` for the snapshot representation. It does not by itself grant supervised-prediction eligibility because physiological qualification is a separate gate.

## 4. Reused current interfaces

The implementation reuses rather than replaces the current 1.16 interfaces:

1. `quality_admission.audit_quality()` for probe-level source/readability/QC/computability states;
2. `analysis_sets.build_analysis_sets()` with explicit `required_feature_records` to map scientific modality `cardiopulmonary` to source namespace `mmwave`;
3. comparison-specific sample construction from the exact HR/BR predictor union.

The interface smoke intentionally uses no Q1/Q2 outcome and trains no model. It verifies only that AVAILABLE HR/BR rows are admitted while unavailable/malformed rows remain explicit structural exclusions. Unrelated Ocular or Movement availability is not referenced.

## 5. Feature handoff boundary

`mmwave_cardiopulmonary_feature_handoff.csv` contains HR and BR provenance in the final semantic shape, but it is **not** the final unified feature registry. It records:

- scientific modality;
- source namespace;
- required devices;
- temporal anchor/scope/status/evidence;
- physiology qualification;
- registry readiness and researcher-freeze requirement.

All supervised prediction eligibility flags remain `false` in this task. Current physiology qualification is `LIMITED_SUPPORTING_ONLY`.

## 6. Outputs

The runner writes:

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

The manifest records governed denominators, key/identity gates, HR/BR finite counts, structural missingness, source snapshot/run provenance, time-legality evidence, scientific modality/device semantics, and the fact that no model or final registry was produced.

## 7. Governed-cohort local run

The 2,320-row snapshot itself is intentionally local-only in the producer contract and is not committed to either GitHub repository. The cloud implementation therefore cannot claim a fresh governed-cohort rerun until the local artifact is executed against this branch.

On the machine that holds the producer snapshot:

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

Do not use `--allow-subset-smoke` for this governed-cohort run. The command fails unless the current Behavior authority and snapshot both conserve exactly 2,320 keys / 116 sessions / 61 groups and the snapshot reproduces 2,180 AVAILABLE + 40 SOURCE_UNAVAILABLE + 100 SOURCE_MALFORMED probes with HR/BR finite exactly 2,180 each.

For a deliberately small schema smoke only, `--allow-subset-smoke` disables the frozen denominator assertions while retaining identity, missingness, and time-legality gates.

## 8. Release interpretation

A successful governed-cohort run supports:

```text
engineering integration qualification = READY
physiology qualification = LIMITED / SUPPORTING_ONLY
```

It does not authorize Cardiopulmonary predictors in the final supervised model. That later decision remains a Formal-method/researcher freeze after the real audit artifacts are reviewed. Under the current evidence, interface readiness and physiological measurement validity remain separate states.
