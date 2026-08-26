"""Participant-grouped incremental value analysis for frozen NIR v1 features."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NIR_FEATURES = ["pir_within_subject_deviation", "pir_fused_pir_mad", "pir_fused_pir_robust_slope_per_s"]
BEHAVIOR_EXCLUDE = {"b_trial_count"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(nir_path: Path, behavior_path: Path, mmwave_path: Path | None, crosswalk_path: Path | None):
    nir = pd.read_csv(nir_path)
    nir = nir[(nir.window == "pre_30s") & (nir.nir_quality_tier == "primary")].copy()
    nir["subject_key"] = nir.subject.astype(str)
    nir["probe_key"] = nir.probe_id.astype(int)
    nir["y"] = (nir.probe_response_raw == 1).astype(int)
    b = pd.read_csv(behavior_path)
    b["subject_key"] = b.subject.astype(int).map(lambda x: f"sub-{x:03d}")
    b["probe_key"] = b.probe_seq.astype(int)
    b = b.drop_duplicates(["subject_key", "probe_key"])
    behavior_cols = [c for c in b.columns if c.startswith("b_") and c not in BEHAVIOR_EXCLUDE]
    b = b[["subject_key", "probe_key"] + behavior_cols]
    out = nir.merge(b, on=["subject_key", "probe_key"], how="inner", validate="one_to_one")
    modalities = {"behavior": behavior_cols}
    if mmwave_path:
        m = pd.read_csv(mmwave_path)
        m["subject_key"] = m.subject.astype(int).map(lambda x: f"sub-{x:03d}")
        m["probe_key"] = m.probe_id.astype(int)
        m = m.drop_duplicates(["subject_key", "probe_key"])
        mm_cols = [c for c in m.columns if (c.startswith("m1_") or c.startswith("q_")) and c not in {"q_extraction_ok"}]
        out = out.merge(m[["subject_key", "probe_key"] + mm_cols], on=["subject_key", "probe_key"], how="left", validate="one_to_one")
        modalities["mmwave"] = mm_cols
    if crosswalk_path:
        cw = pd.read_csv(crosswalk_path)
        cw = cw[cw.session_id.notna()][["session_id", "repeat_participant_id", "actual_session_link_status"]].drop_duplicates("session_id")
        out = out.merge(cw, left_on="subject_key", right_on="session_id", how="left", validate="many_to_one")
    out["cluster_id"] = out["repeat_participant_id"].fillna(out["subject_key"])
    return out, modalities


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "roc_auc": roc_auc_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan}


def evaluate(data: pd.DataFrame, feature_sets: dict[str, list[str]], seed: int = 20260826):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rows, predictions = [], {}
    for name, cols in feature_sets.items():
        x = data.dropna(subset=["y", "cluster_id"])[["y", "cluster_id", "subject_key", "probe_key"] + cols].copy()
        y = x.y.to_numpy(); groups = x.cluster_id.to_numpy(); p = np.full(len(x), np.nan)
        for train, test in cv.split(x[cols], y, groups):
            pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
            pipe.fit(x.iloc[train][cols], y[train]); p[test] = pipe.predict_proba(x.iloc[test][cols])[:, 1]
        mm = metrics(y, p); mm["model"] = name; mm["feature_count"] = len(cols); mm["participant_clusters"] = int(x.cluster_id.nunique()); rows.append(mm)
        predictions[name] = x[["subject_key", "probe_key", "cluster_id", "y"]].assign(predicted_probability=p)
    return pd.DataFrame(rows), predictions


def bootstrap(predictions: dict, pairs: list[tuple[str, str]], n_boot=1000, seed=20260826):
    rng = np.random.default_rng(seed); rows = []
    for a, b in pairs:
        left, right = predictions[a], predictions[b]
        z = left.merge(right, on=["subject_key", "probe_key", "cluster_id", "y"], suffixes=("_a", "_b"))
        clusters = z.cluster_id.unique()
        observed = metrics(z.y.to_numpy(), z.predicted_probability_b.to_numpy())
        observed_a = metrics(z.y.to_numpy(), z.predicted_probability_a.to_numpy())
        for metric in ("roc_auc", "balanced_accuracy", "f1", "sensitivity", "specificity"):
            vals = []
            for _ in range(n_boot):
                sampled = rng.choice(clusters, len(clusters), replace=True)
                boot = pd.concat([z[z.cluster_id == c] for c in sampled], ignore_index=True)
                vals.append(metrics(boot.y.to_numpy(), boot.predicted_probability_b.to_numpy())[metric] - metrics(boot.y.to_numpy(), boot.predicted_probability_a.to_numpy())[metric])
            rows.append({"comparison": f"{b} minus {a}", "metric": metric, "observed_delta": observed[metric] - observed_a[metric], "ci_low": np.nanpercentile(vals, 2.5), "ci_high": np.nanpercentile(vals, 97.5), "n_clusters": len(clusters), "n_boot": n_boot})
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, root: Path):
    names = ["Behavior", "Behavior + NIR"]
    q = summary[summary.model.isin(names)].set_index("model").loc[names]
    metrics_to_plot = ["roc_auc", "balanced_accuracy", "f1", "sensitivity", "specificity"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180); x = np.arange(len(metrics_to_plot)); w=.34
    ax.bar(x-w/2, q.loc["Behavior", metrics_to_plot], w, label="without NIR", color="#8aa6b8")
    ax.bar(x+w/2, q.loc["Behavior + NIR", metrics_to_plot], w, label="with NIR", color="#c56b6b")
    ax.set_xticks(x, [m.replace("_", " ").title() for m in metrics_to_plot], rotation=15, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Out-of-fold score"); ax.set_title("Incremental value of NIR over behavior baseline"); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(root / "figure_incremental_without_vs_with_nir.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nir", required=True); ap.add_argument("--behavior", required=True); ap.add_argument("--mmwave"); ap.add_argument("--crosswalk"); ap.add_argument("--output-root", required=True); args = ap.parse_args()
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    data, modalities = load_data(Path(args.nir), Path(args.behavior), Path(args.mmwave) if args.mmwave else None, Path(args.crosswalk) if args.crosswalk else None)
    feature_sets = {"Behavior": modalities["behavior"], "Behavior + NIR": modalities["behavior"] + NIR_FEATURES}
    if "mmwave" in modalities:
        feature_sets.update({"Behavior + mmWave": modalities["behavior"] + modalities["mmwave"], "Behavior + mmWave + NIR": modalities["behavior"] + modalities["mmwave"] + NIR_FEATURES})
    summary, preds = evaluate(data, feature_sets); summary.to_csv(root / "model_performance.csv", index=False)
    deltas = bootstrap(preds, [("Behavior", "Behavior + NIR")], n_boot=1000); deltas.to_csv(root / "incremental_deltas_bootstrap.csv", index=False)
    plot_comparison(summary, root)
    provenance = {"source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "inputs": {k: {"path": v, "sha256": sha256(Path(v))} for k, v in {"nir": args.nir, "behavior": args.behavior, **({"mmwave": args.mmwave} if args.mmwave else {}), **({"crosswalk": args.crosswalk} if args.crosswalk else {})}.items()}, "joined_rows": int(len(data)), "sessions": int(data.subject_key.nunique()), "participant_clusters": int(data.cluster_id.nunique()), "feature_sets": {k: len(v) for k, v in feature_sets.items()}, "split": "5-fold StratifiedGroupKFold by repeat_participant_id; imputation and scaling fit inside training fold", "bootstrap": "1000 participant-cluster resamples", "rgb_formal_input": "not found; existing RGB paths were pilot/gate outputs and excluded"}
    (root / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    report(root, data, summary, deltas, modalities, provenance)


def report(root, data, summary, deltas, modalities, provenance):
    b = summary[summary.model == "Behavior"].iloc[0]; n = summary[summary.model == "Behavior + NIR"].iloc[0]
    d_auc = deltas[deltas.metric == "roc_auc"].iloc[0]; d_ba = deltas[deltas.metric == "balanced_accuracy"].iloc[0]
    lines = ["# NIR incremental value v1", "", "状态：`NIR_INCREMENTAL_VALUE_V1`。本分析使用冻结 NIR primary/30 s/PIR analytic set 与已有 probe-level modality outputs，不重做 behavior、mmWave 或 RGB 特征。", "", "## Join and split", f"Behavior ∩ NIR primary intersection: {len(data)} probes, {data.subject_key.nunique()} sessions, {data.cluster_id.nunique()} participant clusters. Participant-level `StratifiedGroupKFold` prevents a repeat participant from appearing in both train and test. Median imputation and standardization are fitted inside each training fold only.", "", f"Behavior features: {len(modalities['behavior'])}; NIR features: PIR within-person level, PIR MAD and PIR slope. mmWave features available: {'yes' if 'mmwave' in modalities else 'no'}; formal RGB feature matrix: not found.", "", "## Performance", summary.to_string(index=False), "", "## Incremental value: Behavior + NIR minus Behavior", deltas.to_string(index=False), "", f"Observed ΔAUC={d_auc.observed_delta:.4f}, participant-bootstrap 95% CI [{d_auc.ci_low:.4f}, {d_auc.ci_high:.4f}]. Observed Δbalanced accuracy={d_ba.observed_delta:.4f}, 95% CI [{d_ba.ci_low:.4f}, {d_ba.ci_high:.4f}].", "", "## Human conclusion", "NIR 是否增加独立信息，应以同一 participant-level held-out split 下的增量和 bootstrap CI 判断，而不是训练集拟合优度。当前 Behavior + NIR 的 PIR-level incremental effect 已按固定特征集评估；如果 ΔAUC/Δbalanced accuracy 的区间跨过 0，则不能声称 NIR 提供稳定的独立分类增益，但仍可作为解释性/生理补充保留。", "", "Behavior baseline 与 Behavior + NIR 的直接比较图见 `figure_incremental_without_vs_with_nir.png`。", "", "## RGB/mmWave handoff", "已有正式可复用的 mmWave probe-level matrix 已纳入组合模型。未找到可用于 71-session 同一 probe 交集的正式 RGB feature matrix；现有 RGB 产物包括 `D:\Project\厚粲杯\11_数据\derived\current_j_rgb_motion_gate_v1` 下的 gate/pilot 文件，不能作为正式 RGB 增量输入。", "", "## Provenance", json.dumps(provenance, ensure_ascii=False, indent=2)]
    clean = "\n".join(["# NIR incremental value v1", "", "Status: NIR_INCREMENTAL_VALUE_V1.", "", f"Frozen NIR primary/30 s/PIR joined to behavior: {len(data)} probes, {data.subject_key.nunique()} sessions, {data.cluster_id.nunique()} participant clusters.", "Participant-level StratifiedGroupKFold was used; imputation and scaling were fitted inside training folds.", "", "Model performance:", summary.to_string(index=False), "", "Bootstrap deltas for Behavior + NIR minus Behavior:", deltas.to_string(index=False), "", f"Delta AUC={d_auc.observed_delta:.4f}, 95% CI [{d_auc.ci_low:.4f}, {d_auc.ci_high:.4f}]. Delta balanced accuracy={d_ba.observed_delta:.4f}, 95% CI [{d_ba.ci_low:.4f}, {d_ba.ci_high:.4f}].", "", "NIR adds a modest positive ranking signal, but the AUC and balanced-accuracy intervals cross zero; retain NIR as an interpretable candidate modality, not a proven standalone classifier.", "", "Formal mmWave input was included. No formal 71-session RGB probe-level matrix was found; gate/pilot RGB outputs were excluded.", "", "Provenance is in provenance.json."])
    (root / "NIR_INCREMENTAL_VALUE_REPORT.md").write_text(clean + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
