from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .behavior.evidence import probe_evidence, rolling_evidence
from .behavior.extract import block_metrics, extract_trials
from .behavior.reporting import generate_phase1, generate_phase2, generate_phase3, generate_phase4, load_cohort, phase1_tables, phase2_tables, phase3_tables, phase4_tables
from .config import Config, gate_allows, load_config
from .metadata import run_metadata
from .nir.benchmark import evaluate_all, evaluate_tuned, run_benchmark, run_tuned_benchmark, write_report, write_tuned_report
from .nir.review import build_gate1_review, build_preview, build_representative_review
from .nir.sequence import build_sequences, gate_calibration, run_sequence_detect
from .nir.sequence_eval import evaluate_sequences, write_sequence_report
from .validation import validate_all


class ApprovalGateRequired(RuntimeError):
    pass


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _subject(config: Config, requested: str | None) -> str:
    if not requested:
        raise ValueError("此命令需要 --subject")
    if requested not in config.data["subjects"]["include"]:
        raise ValueError(f"未知被试: {requested}")
    return requested


def _formal_write_guard(config: Config) -> None:
    gate = int(config.section("pipeline").get("approval_gate", 0))
    if gate < 2:
        raise ApprovalGateRequired(
            "approval_gate_required: 审批门 1 禁止写正式 output-v2；请先审阅骨架与 12 眼样本预览。"
        )


def _write_behavior(config: Config, subject: str, tables: dict[str, object]) -> list[str]:
    _formal_write_guard(config)
    output = config.path_value("output_root") / "040-behavior"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = output / f"{subject}_{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(str(path))
    manifest = output.parent / "090-manifests" / f"{subject}_behavior.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(run_metadata(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(str(manifest))
    return written


def cmd_validate(config: Config) -> dict:
    result = validate_all(config)
    result["metadata"] = run_metadata(config)
    return result


def cmd_behavior(config: Config, action: str, subject: str | None, write: bool, phase: int = 1) -> dict:
    if action == "report":
        generators = {1: generate_phase1, 2: generate_phase2, 3: generate_phase3, 4: generate_phase4}
        if phase not in generators:
            raise ValueError(f"行为报告阶段 {phase} 尚未实现")
        if write:
            _formal_write_guard(config)
            return generators[phase](config)
        cohort = load_cohort(config)
        tables = (
            phase1_tables(config, cohort) if phase == 1
            else phase2_tables(cohort) if phase == 2
            else phase3_tables(config, cohort) if phase == 3
            else phase4_tables(config, cohort)
        )
        return {
            "action": "report",
            "phase": phase,
            "dry_run": True,
            "subjects": cohort["subject"].nunique(),
            "trials": len(cohort),
            "tables": {name: {"rows": len(table), "columns": list(table.columns)} for name, table in tables.items()},
            "attention_score_created": False,
        }
    if subject is None:
        raise ValueError("extract/evidence 需要 --subject")
    trials = extract_trials(config, subject)
    if action == "extract":
        tables = {"trials": trials, "blocks": block_metrics(trials)}
    elif action == "evidence":
        tables = {
            "evidence_windows": rolling_evidence(config, trials),
            "probe_evidence": probe_evidence(config, trials),
        }
    else:
        raise ValueError(action)
    result = {
        "subject": subject,
        "action": action,
        "dry_run": not write,
        "tables": {name: {"rows": len(table), "columns": list(table.columns)} for name, table in tables.items()},
        "attention_score_created": False,
    }
    if write:
        result["written"] = _write_behavior(config, subject, tables)
    return result


def _blocked_nir(config: Config, stage: str) -> None:
    if not gate_allows(config, f"nir_{stage}"):
        raise ApprovalGateRequired(
            f"approval_gate_required: nir {stage} 在 configs/preexperiment.yaml 中关闭；"
            "先完成人工真值与对应审批。"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="attention-pipeline")
    parser.add_argument("--config", default="configs/preexperiment.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")

    behavior = commands.add_parser("behavior")
    behavior_actions = behavior.add_subparsers(dest="behavior_action", required=True)
    for action in ("extract", "evidence"):
        item = behavior_actions.add_parser(action)
        item.add_argument("--subject", required=True)
        item.add_argument("--write", action="store_true", help="写正式 output-v2；受审批门保护")
    report = behavior_actions.add_parser("report")
    report.add_argument("--phase", type=int, choices=(1, 2, 3, 4), default=1)
    report.add_argument("--write", action="store_true", help="写正式 output-v2；受审批门保护")

    nir = commands.add_parser("nir")
    nir_actions = nir.add_subparsers(dest="nir_action", required=True)
    review = nir_actions.add_parser("build-review")
    review.add_argument("--preview-eyes", type=int, default=12)
    review.add_argument("--set", choices=("preview", "gate1", "representative"), default="preview", help="preview=均匀抽样预览；gate1=12帧/24眼审批门1审批包；representative=264帧/528眼阶段3代表性集")
    review.add_argument("--force", action="store_true", help="覆盖已存在的审批门1目录")
    benchmark = nir_actions.add_parser("benchmark")
    benchmark.add_argument("--smoke", action="store_true", help="冒烟：只跑配置的 smoke_eyes 只眼验证适配器")
    benchmark.add_argument("--force", action="store_true", help="覆盖已存在的基准目录")
    benchmark.add_argument("--tuned", action="store_true", help="阶段4b：按参数调优网格跑 sweep（18 配置×480 眼 raw）")
    evaluate = nir_actions.add_parser("evaluate")
    evaluate.add_argument("--tuned", action="store_true", help="评估 tuned sweep 并生成 tuned 报告")
    sequence = nir_actions.add_parser("sequence")
    sequence_actions = sequence.add_subparsers(dest="sequence_action", required=True)
    seq_build = sequence_actions.add_parser("build")
    seq_build.add_argument("--force", action="store_true")
    seq_detect = sequence_actions.add_parser("detect")
    seq_detect.add_argument("--force", action="store_true")
    seq_eval = sequence_actions.add_parser("evaluate")
    for action in ("extract", "report"):
        item = nir_actions.add_parser(action)
        item.add_argument("--subject")

    run = commands.add_parser("run")
    run.add_argument("--stage", required=True, choices=("validate", "behavior-extract", "behavior-evidence", "behavior-report", "nir-review-preview"))
    run.add_argument("--subject")
    run.add_argument("--preview-eyes", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate":
            result = cmd_validate(config)
        elif args.command == "behavior":
            subject = None if args.behavior_action == "report" else _subject(config, args.subject)
            result = cmd_behavior(config, args.behavior_action, subject, args.write, getattr(args, "phase", 1))
        elif args.command == "nir":
            if args.nir_action == "build-review":
                if not gate_allows(config, "nir_build_review_preview"):
                    raise ApprovalGateRequired("approval_gate_required: NIR 预览已关闭")
                review_set = getattr(args, "set", "preview")
                if review_set == "gate1":
                    html, review, output_dir = build_gate1_review(config, None, args.force)
                    result = {
                        "preview": True,
                        "set": "gate1",
                        "raw_video_frames": review[["source_video", "avi_frame_idx"]].drop_duplicates().shape[0],
                        "eye_samples": len(review),
                        "output_dir": str(output_dir),
                        "review_html": str(html),
                        "status_counts": review["pipeline_status"].value_counts().to_dict(),
                    }
                elif review_set == "representative":
                    html, review, output_dir = build_representative_review(config, None, args.force)
                    result = {
                        "preview": True,
                        "set": "representative",
                        "raw_video_frames": review[["source_video", "avi_frame_idx"]].drop_duplicates().shape[0],
                        "eye_samples": len(review),
                        "output_dir": str(output_dir),
                        "review_html": str(html),
                        "status_counts": review["pipeline_status"].value_counts().to_dict(),
                    }
                else:
                    html, review = build_preview(config, args.preview_eyes)
                    result = {
                        "preview": True,
                        "set": "preview",
                        "eye_samples": len(review),
                        "raw_video_frames": review[["source_video", "avi_frame_idx"]].drop_duplicates().shape[0],
                        "review_html": str(html),
                        "status_counts": review["pipeline_status"].value_counts().to_dict(),
                    }
            elif args.nir_action == "benchmark":
                _blocked_nir(config, "benchmark")
                if getattr(args, "tuned", False):
                    result = run_tuned_benchmark(config, None, args.force)
                else:
                    result = run_benchmark(config, None, args.smoke, args.force)
            elif args.nir_action == "evaluate":
                _blocked_nir(config, "evaluate")
                if getattr(args, "tuned", False):
                    summary, context = evaluate_tuned(config, None)
                    report = write_tuned_report(config, None, summary, context)
                else:
                    summary, context = evaluate_all(config, None)
                    report = write_report(config, None, summary, context)
                result = {
                    "evaluate": True,
                    "tuned": getattr(args, "tuned", False),
                    "summary": summary.to_dict(orient="records"),
                    "report": str(report),
                }
            elif args.nir_action == "sequence":
                if args.sequence_action == "build":
                    _blocked_nir(config, "sequence_build")
                    result = build_sequences(config, None, args.force)
                elif args.sequence_action == "detect":
                    _blocked_nir(config, "sequence_detect")
                    result = run_sequence_detect(config, None, args.force)
                else:
                    _blocked_nir(config, "sequence_evaluate")
                    summary, context = evaluate_sequences(config, None, None)
                    report = write_sequence_report(config, None, summary, context, None)
                    result = {
                        "evaluate": True,
                        "stage": "sequence",
                        "summary": summary.to_dict(orient="records"),
                        "report": str(report),
                    }
            else:
                _blocked_nir(config, args.nir_action)
                result = {}
        elif args.command == "run":
            if args.stage == "validate":
                result = cmd_validate(config)
            elif args.stage.startswith("behavior-"):
                action = args.stage.removeprefix("behavior-")
                subject = None if action == "report" else _subject(config, args.subject)
                result = cmd_behavior(config, action, subject, False)
            else:
                html, review = build_preview(config, args.preview_eyes)
                result = {"review_html": str(html), "eye_samples": len(review)}
        else:
            raise ValueError(args.command)
        _print(result)
        return 0 if result.get("ok", True) else 2
    except (ApprovalGateRequired, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

