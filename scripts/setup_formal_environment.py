from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS_EXAMPLE = REPO_ROOT / "configs" / "paths.example.yaml"
PATHS_LOCAL = REPO_ROOT / "configs" / "paths.local.yaml"


@dataclass(frozen=True)
class EnvironmentSpec:
    analysis: str
    env_name: str
    yaml_path: Path
    import_check: str
    description: str


ENVIRONMENTS: dict[str, EnvironmentSpec] = {
    "behavior": EnvironmentSpec(
        analysis="behavior",
        env_name="attention-behavior-formal",
        yaml_path=REPO_ROOT / "environments" / "behavior-formal.yml",
        import_check="import numpy,pandas,scipy,statsmodels,yaml; print('behavior formal environment OK')",
        description="Behavior / questionnaire / SART formal downstream",
    ),
    "nir": EnvironmentSpec(
        analysis="nir",
        env_name="attention-nir-formal",
        yaml_path=REPO_ROOT / "environments" / "nir-pupil-formal.yml",
        import_check="import numpy,pandas,pyarrow,scipy,statsmodels,matplotlib,yaml; print('NIR pupil-only formal environment OK')",
        description="NIR pupil-only downstream; no YOLO/RITnet producer dependencies",
    ),
    "rgb": EnvironmentSpec(
        analysis="rgb",
        env_name="attention-rgb-formal",
        yaml_path=REPO_ROOT / "environments" / "rgb-formal.yml",
        import_check="import numpy,pandas,pyarrow,scipy,statsmodels,matplotlib,yaml; print('RGB formal downstream environment OK')",
        description="RGB preserved-output downstream; no Py-Feat/LibreFace/MediaPipe/GPU producer dependencies",
    ),
}


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, text=True, check=check)


def _conda_executable() -> str:
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "未找到 conda。请先安装 Miniconda/Anaconda，并确保 conda 可在当前终端调用。"
        )
    return conda


def _existing_environment_names(conda: str) -> set[str]:
    result = subprocess.run(
        [conda, "env", "list", "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    names: set[str] = set()
    for raw in payload.get("envs", []):
        path = Path(str(raw))
        names.add(path.name)
    return names


def _contains_unresolved_placeholders(path: Path) -> bool:
    if not path.is_file():
        return True
    text = path.read_text(encoding="utf-8")
    return "${" in text or "X:/..." in text or "<REPO_PARENT>" in text


def ensure_local_paths_template(*, create_if_missing: bool) -> tuple[bool, str]:
    if PATHS_LOCAL.is_file():
        if _contains_unresolved_placeholders(PATHS_LOCAL):
            return False, (
                "configs/paths.local.yaml 仍含占位符。请只在该文件填写本机真实路径；"
                "不要修改 science config 去适配某台电脑。"
            )
        return True, "paths.local.yaml exists and has no obvious placeholders"

    if not create_if_missing:
        return False, "configs/paths.local.yaml 不存在"
    if not PATHS_EXAMPLE.is_file():
        raise FileNotFoundError(PATHS_EXAMPLE)
    shutil.copy2(PATHS_EXAMPLE, PATHS_LOCAL)
    return False, (
        "已从 configs/paths.example.yaml 创建 configs/paths.local.yaml。"
        "现在必须填写本机真实路径后再运行正式分析。"
    )


def create_or_update_environment(spec: EnvironmentSpec, *, update: bool) -> None:
    conda = _conda_executable()
    if not spec.yaml_path.is_file():
        raise FileNotFoundError(spec.yaml_path)

    existing = _existing_environment_names(conda)
    if spec.env_name in existing:
        if update:
            _run([conda, "env", "update", "-n", spec.env_name, "-f", str(spec.yaml_path), "--prune"])
        else:
            print(f"环境已存在，保留不重建: {spec.env_name}")
    else:
        _run([conda, "env", "create", "-f", str(spec.yaml_path)])

    # Install the checked-out repository into the selected formal environment.
    # pyarrow is deliberately included in NIR/RGB import checks because the
    # package dependency is installed by this editable install and missing Arrow
    # support should fail during bootstrap rather than during a later data read.
    _run([conda, "run", "-n", spec.env_name, "python", "-m", "pip", "install", "-e", "."])
    _run([conda, "run", "-n", spec.env_name, "python", "-c", spec.import_check])


def print_next_steps(spec: EnvironmentSpec, paths_ready: bool, paths_message: str) -> None:
    print("\n=== formal environment bootstrap summary ===")
    print(f"analysis: {spec.analysis}")
    print(f"environment: {spec.env_name}")
    print(f"purpose: {spec.description}")
    print(f"paths: {paths_message}")
    print("\nNext terminal commands:")
    print(f"  conda activate {spec.env_name}")
    print("  $env:ATTENTION_ANALYSIS_PATHS_CONFIG = (Resolve-Path 'configs/paths.local.yaml').Path  # PowerShell")
    if not paths_ready:
        print("\nBLOCKED FOR FORMAL RUN: edit configs/paths.local.yaml first.")
    else:
        print("\nEnvironment bootstrap complete. Formal execution still requires targeted pytest and representative smoke.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify one isolated formal-analysis Conda environment. "
            "Machine-specific data paths remain exclusively in configs/paths.local.yaml."
        )
    )
    parser.add_argument("analysis", choices=tuple(ENVIRONMENTS), help="Formal downstream to prepare")
    parser.add_argument(
        "--update",
        action="store_true",
        help="If the named environment already exists, update it from the YAML and prune obsolete packages.",
    )
    parser.add_argument(
        "--no-create-path-template",
        action="store_true",
        help="Do not copy configs/paths.example.yaml when paths.local.yaml is missing.",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Only inspect/create configs/paths.local.yaml; do not call conda.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = ENVIRONMENTS[args.analysis]
    paths_ready, paths_message = ensure_local_paths_template(
        create_if_missing=not args.no_create_path_template
    )

    if not args.paths_only:
        create_or_update_environment(spec, update=bool(args.update))

    print_next_steps(spec, paths_ready, paths_message)
    return 0 if paths_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
