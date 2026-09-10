"""Issue #30: read-only, label-blind admission from existing probe products.

No model imports or upstream inference. The behavior table owns the denominator.
Admission is input readiness, not physiological validation or model performance.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import KEY_COLUMNS, PRIMARY_WINDOW, _INPUT_PATHS, load_modality_table

VERSION = "formal-quality-admission-v1"
KEYS = list(KEY_COLUMNS)
GROUP = "participant_group_id"


def validate_omission_inputs(features):
    """Reject alias duplication and total/component algebraic redundancy."""
    names = set(features)
    totals = {"omission_rate", "raw_go_omission_rate"}
    parts = {"clean_go_omission_rate", "timing_ambiguous_go_omission_rate"}
    if totals <= names or (names & totals and parts <= names):
        raise ValueError("duplicate omission inputs: total = clean + timing_ambiguous")
    if len(features) != len(names):
        raise ValueError("duplicate feature names")


def _true(series):
    return series.astype("string").str.lower().isin(["true", "1", "1.0"])


def _validate_keys(frame, modality):
    if not set(KEYS + [GROUP]) <= set(frame):
        raise ValueError(f"{modality}: missing key/identity columns")
    if frame[KEYS + [GROUP]].isna().any().any():
        raise ValueError(f"{modality}: null key/identity")
    if frame[KEYS + [GROUP]].astype(str).apply(lambda s: s.str.strip().isin(["", "nan", "None"])).any().any():
        raise ValueError(f"{modality}: empty key/identity")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{modality}: duplicate probe keys")
    if frame.groupby("session_id")[GROUP].nunique().gt(1).any():
        raise ValueError(f"{modality}: session identity conflict")
    probe = pd.to_numeric(frame[KEYS[-1]], errors="coerce")
    if not (frame.block_id.isin(["b1", "b2"]) & probe.gt(0) & probe.mod(1).eq(0)).all():
        raise ValueError(f"{modality}: illegal formal probe keys")


def _source(frame, modality, matched):
    """Only explicit source evidence, never finite candidate values as source."""
    if modality == "behavior":
        return matched, matched, "formal_behavior_record"
    if modality == "mmwave" and {"mmwave_observed", "mmwave_loadable"} <= set(frame):
        return matched & _true(frame.mmwave_observed), matched & _true(frame.mmwave_loadable), "mmwave_observed+mmwave_loadable"
    if modality == "rgb" and "rgb_source_status" in frame:
        valid = matched & frame.rgb_source_status.isin(["ok", "partial_no_blink"])
        return valid, valid, "rgb_source_status"
    if modality == "nir" and "n_nir_rows" in frame:
        valid = matched & pd.to_numeric(frame.n_nir_rows, errors="coerce").gt(0)
        return valid, valid, "n_nir_rows>0 (record count, not a QC threshold)"
    if "source_present" in frame:
        present = matched & _true(frame.source_present)
        readable = present & _true(frame.source_readable) if "source_readable" in frame else present
        return present, readable, "explicit_source_present"
    return matched & False, matched & False, "source_evidence_missing"


def _aligned(frame, matched):
    valid = matched.copy()
    if "window_name" in frame:
        valid &= frame.window_name.eq(PRIMARY_WINDOW)
    for col in ("anchor_trial_excluded", "anchoring_probe_trial_excluded"):
        if col in frame:
            valid &= _true(frame[col])
    if "window_crosses_block" in frame:
        valid &= frame.window_crosses_block.notna() & ~_true(frame.window_crosses_block)
    if "window_seconds_nominal" in frame:
        valid &= pd.to_numeric(frame.window_seconds_nominal, errors="coerce").eq(30)
    for start, end, anchor in (("window_start_ms", "window_end_ms", "probe_onset_ms"),
                               ("window_start_unix_ms", "window_end_unix_ms", "probe_onset_unix_ms")):
        if {start, end, anchor} <= set(frame):
            s, e, a = (pd.to_numeric(frame[c], errors="coerce") for c in (start, end, anchor))
            valid &= s.lt(e) & e.le(a) & e.sub(s).eq(30000)
    if {"window_effective_start_unix_ms", "block_start_unix_ms", "window_end_unix_ms", "block_end_unix_ms"} <= set(frame):
        exported = frame.block_start_unix_ms.notna() | frame.block_end_unix_ms.notna()
        # E-batch explicitly inherits the formal behavior window; block bounds
        # are not exported there. Do not invent boundaries or reject the batch.
        inherited = frame.get("window_boundary_source", pd.Series("", index=frame.index)).eq("probe_primary_30s")
        valid &= ((~exported & inherited) | (frame.window_effective_start_unix_ms.ge(frame.block_start_unix_ms) & frame.window_end_unix_ms.le(frame.block_end_unix_ms)))
    return valid.fillna(False)


def _native(frame, modality, feature):
    valid = pd.Series(True, index=frame.index)
    evidence = []
    if "native_qc_valid" in frame:
        valid &= _true(frame.native_qc_valid)
        evidence.append("native_qc_valid")
    col = None
    if modality == "behavior":
        col = {"go_correct_rt_cv": "rt_cv_status", "go_correct_rt_theilsen_slope_ms_per_s": "rt_slope_status",
               "dprime_loglinear": "sdt_status", "criterion_c": "sdt_status", "beta": "sdt_status"}.get(feature)
    if col and col in frame:
        valid &= frame[col].eq("estimable")
        evidence.append(col)
    if modality == "nir" and f"{feature}_n_valid" in frame:
        col = f"{feature}_n_valid"
        valid &= pd.to_numeric(frame[col], errors="coerce").gt(0)
        evidence.append(col)
    if modality == "rgb" and feature.startswith("blink_") and "rgb_source_status" in frame:
        valid &= frame.rgb_source_status.eq("ok")
        evidence.append("rgb_source_status:blink")
    status = pd.Series("not_explicitly_gated", index=frame.index)
    if evidence:
        status[:] = np.where(valid, "valid", "invalid")
    return valid.fillna(False), status, ";".join(evidence) or "upstream_finite_value_semantics"


def audit_quality(tables, feature_blocks, rules=None, comparison_sets=None):
    rules = rules or {}
    validate_omission_inputs(feature_blocks.get("behavior", []))
    features = {k: list(v) for k, v in feature_blocks.items() if k != "nir_primary"}
    if "nir_primary" in feature_blocks:
        features["nir"] = list(feature_blocks["nir_primary"]["metrics"])
    base = tables["behavior"].reset_index(drop=True)
    _validate_keys(base, "behavior")
    if base.empty:
        raise ValueError("empty formal behavior denominator")
    identity = base[KEYS + [GROUP]].copy()
    audits, coverage, availability, redundant = [], [], [], []
    masks = {}
    input_rows = {}
    for modality, cols in features.items():
        source = tables.get(modality, pd.DataFrame())
        input_rows[modality] = {"rows": len(source), "outside_formal_probe_n": 0}
        if source.empty:
            source = pd.DataFrame(columns=KEYS + [GROUP])
        else:
            _validate_keys(source, modality)
            check = source.merge(identity, on=KEYS, suffixes=("", "_formal"), how="left", validate="one_to_one")
            input_rows[modality]["outside_formal_probe_n"] = int(check[f"{GROUP}_formal"].isna().sum())
            if (check[f"{GROUP}_formal"].notna() & check[GROUP].ne(check[f"{GROUP}_formal"])).any():
                raise ValueError(f"{modality}: identity disagrees with behavior")
        frame = identity.merge(source.drop(columns=[GROUP]), on=KEYS, how="left", indicator=True, validate="one_to_one")
        matched = frame._merge.eq("both")
        present, readable, source_basis = _source(frame, modality, matched)
        aligned = _aligned(frame, matched)
        opportunity = present & readable & aligned
        count = int(opportunity.sum())
        availability.append(dict(modality=modality, formal_probe_n=len(base), source_present_n=int(present.sum()),
                                 source_readable_n=int(readable.sum()), source_aligned_probe_n=count,
                                 modality_availability_rate=count / len(base), source_evidence=source_basis))
        values = {}
        admitted_masks = []
        for feature in cols:
            qc, status, qc_basis = _native(frame, modality, feature)
            raw = pd.to_numeric(frame[feature], errors="coerce") if feature in frame else pd.Series(np.nan, index=frame.index)
            finite = pd.Series(np.isfinite(raw), index=frame.index)
            computable = opportunity & qc & finite
            reason = np.select([~matched, ~present, ~readable, ~aligned, ~qc,
                                pd.Series(feature not in source, index=frame.index), ~finite],
                               ["record_missing", "source_missing_or_unverified", "source_unreadable", "alignment_invalid",
                                "native_qc_invalid", "column_missing", "nonfinite_value"], default="computable")
            detail = identity.copy()
            detail["modality"], detail["feature"] = modality, feature
            detail["source_present"], detail["source_readable"], detail["aligned"] = present, readable, aligned
            detail["native_qc_valid"], detail["native_qc_evidence"] = status, qc_basis
            detail["source_evidence"] = source_basis
            detail["alignment_evidence"] = frame.get("window_boundary_source", "formal_probe_product_and_exported_window_checks")
            detail["feature_computable"], detail["inclusion_reason"] = computable, reason
            detail["value"] = raw.where(computable)
            audits.append(detail)
            x = raw[computable]
            n, unique = len(x), int(x.nunique())
            rate = n / count if count else None
            floor = bool(n and feature in rules.get("proportion_features", []) and x.le(.02).mean() >= .95)
            ceiling = bool(n and feature in rules.get("proportion_features", []) and x.ge(.98).mean() >= .95)
            failures = []
            if feature not in source: failures.append("column_missing")
            if not n: failures.append("all_missing")
            if unique == 1: failures.append("zero_variance")
            if unique < 3: failures.append("fewer_than_3_unique_values")
            if rate is None or rate < .80: failures.append("coverage_below_0.80")
            admitted = not failures
            detail["feature_main_candidate"] = admitted
            detail["native_source_reason"] = frame.get("mmwave_missing_reason", frame.get("rgb_source_status", ""))
            coverage.append(dict(modality=modality, feature=feature, formal_probe_n=len(base), opportunity_n=count,
                                 computable_n=n, modality_availability_rate=count / len(base),
                                 feature_computable_coverage=rate, overall_effective_rate=n / len(base),
                                 unique_valid_n=unique, zero_variance=unique == 1, severe_floor=floor,
                                 severe_ceiling=ceiling, review_required=floor or ceiling,
                                 main_candidate=admitted, admission_reason=";".join(failures) or "passed_input_quality",
                                 native_qc_evidence=qc_basis))
            values[feature] = raw.where(computable)
            if admitted: admitted_masks.append(computable)
        # Explicit complete-feature set; no implicit whole-modality imputation.
        masks[modality] = (pd.concat(admitted_masks, axis=1).all(axis=1) if admitted_masks else opportunity & False)
        for family, members in rules.get("measurement_families", {}).items():
            for a, b in combinations([c for c in members if c in values], 2):
                pair = pd.DataFrame({a: values[a], b: values[b]}).dropna()
                if len(pair) < 3 or pair.nunique().min() < 2: continue
                rho = float(pair.corr(method="spearman").loc[a, b])
                if abs(rho) >= .90:
                    redundant.append(dict(modality=modality, family=family, feature_a=a, feature_b=b,
                                          paired_n=len(pair), spearman_rho=rho, action="review_only"))
    sets = []
    for name, modalities in (comparison_sets or {"all_modalities": list(features)}).items():
        if not modalities or set(modalities) - set(masks):
            raise ValueError(f"unknown or empty comparison set: {name}")
        included = pd.concat([masks[m] for m in modalities], axis=1).all(axis=1)
        rows = identity.copy()
        rows["analysis_set_id"] = name
        rows["included"] = included
        rows["set_rule"] = "all_admitted_features_computable_in_each_required_modality"
        sets.append(rows)
    return {"probe_quality": pd.concat(audits, ignore_index=True), "feature_admission": pd.DataFrame(coverage),
            "modality_availability": pd.DataFrame(availability), "analysis_sets": pd.concat(sets, ignore_index=True),
            "redundancy_review": pd.DataFrame(redundant, columns=["modality", "family", "feature_a", "feature_b", "paired_n", "spearman_rho", "action"]),
            "input_rows": input_rows}


def run_quality_admission(config_path, *, paths_config=None, run_id):
    from attention_pipeline.config import load_config
    config = load_config(config_path, paths_config=paths_config)
    root = config.path_value("data_root")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name")
    output = config.path_value("output_root") / run_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    aliases = {"behavior": "behavior_probe", "nir": "nir_probe_model_table", "rgb": "rgb_probe_pre30s",
               "mmwave": "mmwave_probe_merge_ready", "mmwave_e": "mmwave_probe_merge_ready_e", "bridge": "mmwave_identity_bridge"}
    input_paths = {key: config.data.get("inputs", {}).get(aliases[key], rel) for key, rel in _INPUT_PATHS.items()}
    bridge_path = root / input_paths["bridge"]
    bridge = pd.read_csv(bridge_path) if bridge_path.is_file() else None
    if bridge is not None and (bridge.session_id.isna().any() or bridge.session_id.duplicated().any()):
        raise ValueError("identity bridge has null/duplicate session keys")
    tables, problems = {}, []
    for modality in ("behavior", "nir", "mmwave", "rgb"):
        tables[modality], issues = load_modality_table(root, modality, bridge, input_paths=input_paths)
        problems.extend(issues)
    fatal = [p for p in problems if not p.startswith("missing_input:")]
    if fatal: raise ValueError(f"invalid input schema: {fatal}")
    result = audit_quality(tables, config.section("feature_blocks"), config.section("quality_admission"), config.section("combinations"))
    destination = output / "quality_admission"
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            path = destination / f"{name}.csv"
            value.to_csv(path, index=False, encoding="utf-8-sig")
            files[name] = str(path)
    admitted = result["feature_admission"]
    candidate_blocks = {}
    for modality, group in admitted[admitted.main_candidate].groupby("modality"):
        candidate_blocks[modality] = group.feature.tolist()
    (destination / "candidate_features.json").write_text(json.dumps(candidate_blocks, indent=2) + "\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[3]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip())
    summary = dict(status="PASS", rule_version=VERSION, run_id=run_id, code_sha=sha, code_dirty=dirty,
                   coverage_threshold=.80, min_unique_values=3, floor_ceiling_fraction=.95,
                   floor_cutoff=.02, ceiling_cutoff=.98, same_family_redundancy_abs_rho=.90,
                   rules=config.section("quality_admission"), formal_probe_n=len(tables["behavior"]),
                   denominator="formal behavior probe table, never modality intersection",
                   input_rows=result["input_rows"], input_problems=problems,
                   input_paths=input_paths,
                   input_sha256={key: hashlib.sha256((root / rel).read_bytes()).hexdigest() if (root / rel).is_file() else None for key, rel in input_paths.items()},
                   config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                   candidates=candidate_blocks, output_files=files, models_trained=False,
                   limitations=["Input quality admission is not scientific release; mmWave physiological thresholds remain unfrozen.",
                                "No imputation; analysis sets require every admitted feature to be computable.",
                                "Global input inventory is descriptive; learned preprocessing/selection must be refitted within training folds.",
                                "Time checks reuse exported boundaries; frame-level upstream processing is not re-audited."])
    summary["analysis_sets"] = {}
    for name, rows in result["analysis_sets"].groupby("analysis_set_id"):
        selected = rows[rows.included]
        summary["analysis_sets"][name] = {"probe_n": len(selected), "session_n": int(selected.session_id.nunique()),
                                           "participant_group_n": int(selected[GROUP].nunique())}
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
