from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.formal_analysis.cohort import canonical_session_id
from attention_pipeline.formal_analysis.identity_questionnaire import load_repeat_registry

PIPELINE_VERSION = "rgb-formal-downstream-v1"
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
CONTEXT_COLUMNS = (
    "phase", "block", "trial_num", "cycle_num", "position_in_cycle", "is_no_go",
    "response", "correct", "commission", "omission", "is_probe", "probe_response",
    "probe_vigilance", "absolute_onset_time", "probe_onset_time", "behavior_state",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path(config: Config, name: str) -> Path:
    return config.path_value(name)


def discover_sessions(raw_root: Path) -> list[str]:
    if not raw_root.is_dir():
        return []
    return sorted(
        canonical_session_id(p.name)
        for p in raw_root.iterdir()
        if p.is_dir() and p.name.lower().startswith("sub-")
    )


def _subject_dir(raw_root: Path, session_id: str) -> Path:
    exact = raw_root / session_id
    if exact.is_dir():
        return exact
    matches = [p for p in raw_root.iterdir() if p.is_dir() and canonical_session_id(p.name) == session_id]
    if len(matches) != 1:
        raise FileNotFoundError(f"RGB subject directory unresolved for {session_id}: {matches}")
    return matches[0]


def _find_subject_file(subject_dir: Path, session_id: str, suffix: str) -> Path | None:
    exact = subject_dir / f"{session_id}{suffix}"
    if exact.is_file():
        return exact
    matches = sorted(subject_dir.glob(f"*{suffix}"))
    return matches[0] if len(matches) == 1 else None


def _load_optional(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def _point(frame: pd.DataFrame, idx: int) -> tuple[pd.Series, pd.Series]:
    return _numeric(frame, f"mesh_x_{idx}"), _numeric(frame, f"mesh_y_{idx}")


def _distance(frame: pd.DataFrame, a: int, b: int) -> pd.Series:
    ax, ay = _point(frame, a); bx, by = _point(frame, b)
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _ear(frame: pd.DataFrame, indices: tuple[int, ...]) -> pd.Series:
    p1, p2, p3, p4, p5, p6 = indices
    h = _distance(frame, p1, p4).replace(0, np.nan)
    return (_distance(frame, p2, p6) + _distance(frame, p3, p5)) / (2.0 * h)


def _primary_face(face: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if face.empty:
        return face.copy(), "face_raw_missing"
    current = face.copy()
    if "detected" in current:
        current = current[current["detected"].fillna(False).astype(bool)]
    if "primary_face" in current and current["primary_face"].fillna(False).any():
        return current[current["primary_face"].fillna(False).astype(bool)].copy(), "producer_primary_face"
    if "face_track_id" in current and "video_frame_position" in current:
        occupancy = current.groupby("face_track_id")["video_frame_position"].nunique().sort_values(ascending=False)
        if len(occupancy):
            return current[current["face_track_id"].eq(occupancy.index[0])].copy(), "longest_track_fallback_candidate"
    if "face_rank" in current:
        return current[pd.to_numeric(current["face_rank"], errors="coerce").eq(0)].copy(), "face_rank0_fallback_candidate"
    return current.copy(), "unresolved_multiple_face_candidate"


def _open_reference(values: pd.Series, phase: pd.Series, preferred: str, min_n: int) -> tuple[float, str, int]:
    x = pd.to_numeric(values, errors="coerce")
    baseline = x[phase.astype(str).str.lower().eq(preferred.lower()) & x.notna()]
    if len(baseline) >= min_n:
        source, label = baseline, "baseline_top30_median"
    else:
        source, label = x[x.notna()], "all_valid_top30_median_fallback_not_resting_baseline"
    if source.empty:
        return math.nan, label, 0
    q = source.quantile(0.70)
    return float(source[source >= q].median()), label, int(len(source))


def _blink_events(times: pd.Series, closed: pd.Series, min_ms: float, max_ms: float) -> tuple[pd.Series, pd.DataFrame]:
    t = pd.to_numeric(times, errors="coerce").to_numpy(float)
    c = closed.fillna(False).astype(bool).to_numpy()
    event_id = np.full(len(c), np.nan)
    events: list[dict[str, Any]] = []
    eid = 0; start: int | None = None
    for i, state in enumerate(np.r_[c, False]):
        if state and start is None:
            start = i
        elif not state and start is not None:
            end = i - 1
            duration = float(t[end] - t[start]) if end > start and np.isfinite(t[end]) and np.isfinite(t[start]) else 0.0
            if min_ms <= duration <= max_ms:
                eid += 1; event_id[start:i] = eid
                events.append({"blink_event_id": eid, "start_unix_ms": t[start], "end_unix_ms": t[end], "duration_ms": duration})
            start = None
    return pd.Series(event_id, index=closed.index, dtype="Float64"), pd.DataFrame(events)


def derive_face_features(face: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    face, primary_status = _primary_face(face)
    if face.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "missing", "primary_face_status": primary_status}
    if "unix_ms" not in face:
        raise ValueError("face raw missing unix_ms")
    out = face[[c for c in ("subject", "unix_ms", "video_frame_position", "capture_frame_idx", "phase", "block") if c in face]].copy()
    mesh_ok = all(f"mesh_x_{i}" in face and f"mesh_y_{i}" in face for i in set(RIGHT_EYE + LEFT_EYE))
    if mesh_ok:
        out["ear_right"] = _ear(face, RIGHT_EYE); out["ear_left"] = _ear(face, LEFT_EYE)
        out["ear_mean"] = out[["ear_left", "ear_right"]].mean(axis=1)
        ocfg = config.section("ocular")
        refcfg = ocfg["open_reference"]
        ref, ref_source, ref_n = _open_reference(out["ear_mean"], out.get("phase", pd.Series("", index=out.index)), str(refcfg["preferred_phase"]), int(refcfg["minimum_valid_frames"]))
        out["eye_open_reference"] = ref
        out["eye_openness_norm"] = out["ear_mean"] / ref if np.isfinite(ref) and ref > 0 else np.nan
        out["closure_fraction"] = (1.0 - out["eye_openness_norm"].clip(upper=1.0)).clip(0.0, 1.0)
        threshold = float(ocfg["blink_candidate"]["relative_openness_threshold"])
        out["closure80_candidate"] = out["eye_openness_norm"].le(threshold)
        out["blink_event_id"], blink_events = _blink_events(
            out["unix_ms"], out["closure80_candidate"],
            float(ocfg["blink_candidate"]["minimum_closed_duration_ms"]),
            float(ocfg["blink_candidate"]["maximum_closed_duration_ms"]),
        )
    else:
        blink_events = pd.DataFrame(); ref_source = "mesh_landmarks_missing"; ref_n = 0
    # Preserve actual AU/head/gaze numeric columns without assuming a particular backend schema.
    aux = [c for c in face.columns if re.match(r"^(AU\d+|Pose_|Head|Gaze|gaze|pitch|yaw|roll)", str(c), re.I)]
    for c in aux:
        numeric = pd.to_numeric(face[c], errors="coerce")
        if numeric.notna().any(): out[f"face__{c}"] = numeric
    return out.reset_index(drop=True), blink_events, {
        "status": "complete", "primary_face_status": primary_status, "mesh_ear_available": mesh_ok,
        "open_reference_source": ref_source, "open_reference_source_n": ref_n,
        "blink_threshold_status": "candidate_requires_real_data_freeze",
    }


def derive_motion_features(motion: pd.DataFrame) -> pd.DataFrame:
    if motion.empty: return pd.DataFrame()
    keep = [c for c in ("subject", "unix_ms", "video_frame_position", "capture_frame_idx", *CONTEXT_COLUMNS,
        "dt_ms", "gap_before", "irregular_dt", "motion_valid", "gray_mean", "gray_mean_delta",
        "changed_pixel_ratio", "global_motion_energy", "global_motion_energy_per_sec") if c in motion]
    return motion[keep].copy().sort_values("unix_ms")


def derive_pose_features(pose: pd.DataFrame) -> pd.DataFrame:
    if pose.empty: return pd.DataFrame()
    if "unix_ms" not in pose: raise ValueError("pose raw missing unix_ms")
    name_col = "landmark_name" if "landmark_name" in pose else None
    xcol, ycol = ("x", "y") if {"x", "y"}.issubset(pose.columns) else ("landmark_x", "landmark_y")
    if xcol not in pose or ycol not in pose:
        return pose[[c for c in ("subject", "unix_ms", "phase", "block") if c in pose]].drop_duplicates("unix_ms")
    rows: list[dict[str, Any]] = []
    for t, g in pose.groupby("unix_ms", sort=True):
        row: dict[str, Any] = {"unix_ms": t}
        for c in ("subject", "phase", "block"):
            if c in g: row[c] = g[c].iloc[0]
        vis = pd.to_numeric(g["visibility"], errors="coerce") if "visibility" in g else pd.Series(np.nan, index=g.index)
        row["pose_visibility_mean"] = float(vis.mean()) if vis.notna().any() else np.nan
        if name_col:
            by = {str(r[name_col]).lower(): r for _, r in g.iterrows()}
            for name in ("nose", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"):
                r = by.get(name)
                if r is not None:
                    row[f"pose_{name}_x"] = pd.to_numeric(pd.Series([r[xcol]]), errors="coerce").iloc[0]
                    row[f"pose_{name}_y"] = pd.to_numeric(pd.Series([r[ycol]]), errors="coerce").iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("unix_ms")
    dt = pd.to_numeric(out["unix_ms"], errors="coerce").diff() / 1000.0
    for stem in ("pose_nose", "pose_left_shoulder", "pose_right_shoulder", "pose_left_wrist", "pose_right_wrist"):
        if f"{stem}_x" in out and f"{stem}_y" in out:
            dist = np.sqrt(pd.to_numeric(out[f"{stem}_x"], errors="coerce").diff() ** 2 + pd.to_numeric(out[f"{stem}_y"], errors="coerce").diff() ** 2)
            out[f"{stem}_speed_per_sec"] = dist / dt.replace(0, np.nan)
    if {"pose_left_shoulder_x", "pose_right_shoulder_x", "pose_left_shoulder_y", "pose_right_shoulder_y"}.issubset(out):
        dx = out["pose_right_shoulder_x"] - out["pose_left_shoulder_x"]
        dy = out["pose_right_shoulder_y"] - out["pose_left_shoulder_y"]
        out["shoulder_line_angle_rad"] = np.arctan2(dy, dx)
    return out


def attach_behavior_context(native: pd.DataFrame, motion: pd.DataFrame, tolerance_ms: int = 150) -> pd.DataFrame:
    if native.empty or motion.empty or "unix_ms" not in native or "unix_ms" not in motion: return native.copy()
    context = motion[["unix_ms", *[c for c in CONTEXT_COLUMNS if c in motion]]].drop_duplicates("unix_ms").sort_values("unix_ms")
    left = native.drop(columns=[c for c in CONTEXT_COLUMNS if c in native], errors="ignore").sort_values("unix_ms")
    return pd.merge_asof(left, context, on="unix_ms", direction="nearest", tolerance=tolerance_ms)


def _mad(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna(); return float((x - x.median()).abs().median()) if len(x) else np.nan


def _summarize(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str], scale: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        base = dict(zip(group_cols, key))
        for metric in metric_cols:
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            if not len(x): continue
            rows.append({**base, "scale": scale, "metric": metric, "n_valid": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "sd": float(x.std(ddof=1)) if len(x)>=2 else np.nan, "mad": _mad(x), "q10": float(x.quantile(.1)), "q90": float(x.quantile(.9))})
    return pd.DataFrame(rows)


def build_multiscale(features: pd.DataFrame, probes: pd.DataFrame, probe_windows: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    idcols = set(("subject", "session_id", "participant_group_id", "participant_key", "unix_ms", *CONTEXT_COLUMNS))
    metrics = [c for c in features.columns if c not in idcols and pd.api.types.is_numeric_dtype(features[c]) and c not in {"video_frame_position", "capture_frame_idx", "trial_num", "block", "cycle_num", "absolute_onset_time", "probe_onset_time"}]
    parts = [_summarize(features, ["session_id", "participant_group_id"], metrics, "session")]
    if "block" in features: parts.append(_summarize(features.dropna(subset=["block"]), ["session_id", "participant_group_id", "block"], metrics, "block"))
    if "trial_num" in features: parts.append(_summarize(features.dropna(subset=["block", "trial_num"]), ["session_id", "participant_group_id", "block", "trial_num"], metrics, "trial"))
    summary = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
    probe_rows: list[pd.DataFrame] = []
    if not probes.empty:
        for p in probes.itertuples(index=False):
            for seconds in probe_windows:
                onset = float(p.probe_onset_time); anchor = int(p.trial_num); block = int(p.block)
                w = features[(pd.to_numeric(features.get("block"), errors="coerce").eq(block)) & pd.to_numeric(features["unix_ms"], errors="coerce").ge(onset-seconds*1000) & pd.to_numeric(features["unix_ms"], errors="coerce").lt(onset)]
                if "trial_num" in w: w = w[pd.to_numeric(w["trial_num"], errors="coerce").lt(anchor)]
                current = _summarize(w, ["session_id", "participant_group_id"], metrics, "probe")
                if not current.empty:
                    current["probe_trial_num"] = anchor; current["probe_onset_time"] = onset; current["window_seconds"] = seconds
                    current["q1_nominal_4class"] = getattr(p, "probe_response", np.nan); current["q2_ordinal_4level"] = getattr(p, "probe_vigilance", np.nan)
                    current["anchor_trial_excluded"] = True; probe_rows.append(current)
    return summary, pd.concat(probe_rows, ignore_index=True) if probe_rows else pd.DataFrame()


def candidate_validation(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows=[]
    for (scale, metric), g in summary.groupby(["scale", "metric"], sort=True):
        x=pd.to_numeric(g["median"], errors="coerce"); finite=x.dropna(); pmeans=g.assign(_x=x).groupby("participant_group_id")["_x"].mean().dropna(); centered=g.assign(_x=x).dropna(subset=["_x"]); centered["_w"]=centered["_x"]-centered.groupby("participant_group_id")["_x"].transform("mean")
        rows.append({"scale":scale,"metric":metric,"n_rows":len(g),"n_valid":len(finite),"coverage":len(finite)/len(g) if len(g) else 0,"participant_group_n":g.loc[x.notna(),"participant_group_id"].nunique(),"session_n":g.loc[x.notna(),"session_id"].nunique(),"between_participant_variance":pmeans.var(ddof=1) if len(pmeans)>=2 else np.nan,"within_participant_variance":centered["_w"].var(ddof=1) if len(centered)>=2 else np.nan,"candidate_status":"eligible_candidate" if len(finite)/len(g)>=.8 and finite.nunique()>=3 else "needs_review"})
    validation=pd.DataFrame(rows)
    wide=summary[summary["scale"].eq("block")].pivot_table(index=["participant_group_id","session_id","block"],columns="metric",values="median",aggfunc="first") if "block" in summary else pd.DataFrame()
    corr=wide.corr(method="spearman",min_periods=3) if not wide.empty else pd.DataFrame(); red=[]
    cols=list(corr.columns)
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            r=corr.loc[a,b]; red.append({"metric_a":a,"metric_b":b,"spearman_r":r,"abs_r":abs(r) if np.isfinite(r) else np.nan,"high_redundancy_flag":bool(np.isfinite(r) and abs(r)>=.9)})
    decisions=validation[["scale","metric","candidate_status"]].copy(); decisions["final_endpoint_freeze_status"]="pending_real_data_scientific_review"; decisions["selection_contract"]="coverage + within/between + redundancy + scientific validity; never outcome p-value screening"
    return validation,pd.DataFrame(red),decisions


def _identity_overlay(config: Config, sessions: list[str]) -> pd.DataFrame:
    registry=load_repeat_registry(config, path_key=str(config.section("identity").get("repeat_registry_path_key","repeat_registry")))
    reg=registry[registry["session_id"].isin(sessions)].copy()
    raw_cohort=pd.read_csv(config.registry_path(str(config.section("identity").get("cohort_manifest_path_key","cohort_manifest"))),encoding="utf-8-sig")
    session_col="session_id" if "session_id" in raw_cohort else "subject"
    raw_cohort["session_id"]=raw_cohort[session_col].map(canonical_session_id)
    include=raw_cohort["include"].fillna(False).astype(bool) if "include" in raw_cohort else pd.Series(True,index=raw_cohort.index)
    cohort=raw_cohort[include].copy(); group_col="repeat_participant_id" if "repeat_participant_id" in cohort else None
    base=pd.DataFrame({"session_id":sessions}).merge(reg,on="session_id",how="left",validate="one_to_one")
    if group_col: base=base.merge(cohort[["session_id",group_col]].drop_duplicates(),on="session_id",how="left",validate="one_to_one")
    base["participant_group_id"]=base["participant_key"].astype("string")
    if group_col: base["participant_group_id"]=base["participant_group_id"].fillna(base[group_col].astype("string"))
    base["participant_group_source"]=np.where(base["participant_key"].notna(),"participant_key_registry","governed_cohort_fallback")
    return base


def run_rgb_formal_pipeline(config_path: str | Path, *, subjects: Iterable[str] | None=None, force: bool=False) -> dict[str, Any]:
    config=load_config(config_path); raw_root=_path(config,"raw_root"); ready_root=_path(config,"analysis_ready_root"); output_root=_path(config,"output_root")
    sessions=[canonical_session_id(s) for s in subjects] if subjects else discover_sessions(raw_root)
    if not sessions: raise ValueError("no RGB sessions discovered")
    identity=_identity_overlay(config,sessions); ready_root.mkdir(parents=True,exist_ok=True); output_root.mkdir(parents=True,exist_ok=True)
    all_features=[]; all_probes=[]; failures=[]; session_manifests=[]
    inp=config.section("inputs")
    for session in sessions:
        try:
            sdir=_subject_dir(raw_root,session); face_path=_find_subject_file(sdir,session,str(inp["face_suffix"])); pose_path=_find_subject_file(sdir,session,str(inp["pose_suffix"])); motion_path=_find_subject_file(sdir,session,str(inp["motion_suffix"]))
            face=_load_optional(face_path); pose=_load_optional(pose_path); motion=derive_motion_features(_load_optional(motion_path)); face_d,blinks,face_status=derive_face_features(face,config); pose_d=derive_pose_features(pose)
            face_d=attach_behavior_context(face_d,motion); pose_d=attach_behavior_context(pose_d,motion)
            native=[]
            for modality,frame in (("face",face_d),("pose",pose_d),("motion",motion)):
                if frame.empty: continue
                f=frame.copy(); f["session_id"]=session; f["modality"]=modality; native.append(f)
                target=ready_root/session; target.mkdir(parents=True,exist_ok=True); f.to_parquet(target/f"{session}_{modality}_derived.parquet",index=False)
            combined=pd.concat(native,ignore_index=True,sort=False) if native else pd.DataFrame(); idrow=identity[identity["session_id"].eq(session)]
            if idrow.empty: raise ValueError("session absent from governed identity overlay")
            combined["participant_group_id"]=str(idrow.iloc[0]["participant_group_id"]); combined["participant_key"]=idrow.iloc[0].get("participant_key",pd.NA)
            probes=motion[pd.to_numeric(motion.get("is_probe"),errors="coerce").eq(1) & pd.to_numeric(motion.get("probe_onset_time"),errors="coerce").notna()].drop_duplicates(["block","trial_num","probe_onset_time"]) if not motion.empty else pd.DataFrame()
            all_features.append(combined); all_probes.append(probes.assign(session_id=session))
            session_manifests.append({"session_id":session,"status":"complete","face_file":str(face_path) if face_path else None,"pose_file":str(pose_path) if pose_path else None,"motion_file":str(motion_path) if motion_path else None,"face_status":face_status,"blink_event_n":int(len(blinks)),"participant_group_source":idrow.iloc[0]["participant_group_source"]})
        except Exception as exc:
            failures.append({"session_id":session,"stage":"rgb_analysis_ready","error_type":type(exc).__name__,"error":str(exc)})
    pd.DataFrame(failures).to_csv(output_root/"rgb_failures.csv",index=False,encoding="utf-8-sig")
    if failures:
        manifest={"status":"blocked","pipeline_version":PIPELINE_VERSION,"sessions_requested":len(sessions),"sessions_failed":len(failures),"scientific_inference_authorized":False}; (output_root/"rgb_formal_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); return manifest
    features=pd.concat(all_features,ignore_index=True,sort=False); probes=pd.concat(all_probes,ignore_index=True,sort=False) if all_probes else pd.DataFrame()
    summary,probe_summary=build_multiscale(features,probes,config.section("windows")["probe_pre_seconds"])
    validation,redundancy,decisions=candidate_validation(summary)
    features.to_parquet(ready_root/"rgb_feature_native_long.parquet",index=False); summary.to_csv(output_root/"rgb_multiscale_metrics.csv",index=False,encoding="utf-8-sig"); probe_summary.to_csv(output_root/"rgb_probe_metrics.csv",index=False,encoding="utf-8-sig"); validation.to_csv(output_root/"rgb_candidate_metric_validation.csv",index=False,encoding="utf-8-sig"); redundancy.to_csv(output_root/"rgb_metric_redundancy.csv",index=False,encoding="utf-8-sig"); decisions.to_csv(output_root/"rgb_endpoint_decisions.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(session_manifests).to_json(output_root/"rgb_session_manifest.jsonl",orient="records",lines=True,force_ascii=False)
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"complete","pipeline_version":PIPELINE_VERSION,"sessions":len(sessions),"participant_groups":int(identity["participant_group_id"].nunique()),"rppg_in_scope":False,"expensive_models_rerun":False,"endpoint_freeze":"pending_real_data_scientific_review","strict_preprobe_anchor_exclusion":True,"inference_authorized_by_code_alone":False,"files":{"feature_native_long":"rgb_feature_native_long.parquet","multiscale":"rgb_multiscale_metrics.csv","probe":"rgb_probe_metrics.csv","validation":"rgb_candidate_metric_validation.csv","redundancy":"rgb_metric_redundancy.csv","decisions":"rgb_endpoint_decisions.csv","failures":"rgb_failures.csv"}}; (output_root/"rgb_formal_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); return manifest
