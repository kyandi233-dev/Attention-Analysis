from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from .config import Config


def _literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def validate_protocol(config: Config) -> dict:
    root = config.path_value("formal_experiment_root")
    main = root / "01-MainProgram"
    sart = main / "core" / "sart_task.py"
    sequence_dir = main / "sequences"
    source = sart.read_text(encoding="utf-8")
    # 校验对象是正式实验程序（BBB/注意4分类＋警觉度4点双问题探针）；历史 protocol 由行为阶段单独使用。
    expected = config.section("formal_protocol")
    checks = {
        "block_order": _literal_assignment(source, "BLOCK_ORDER") == expected["block_order"],
        "trials_per_cycle": _literal_assignment(source, "CYCLE_LENGTH") == expected["trials_per_cycle"],
        "cycles_per_block": _literal_assignment(source, "N_CYCLES_FORMAL") == expected["cycles_per_block"],
        "nominal_trial_ms": round((_literal_assignment(source, "STIM_DURATION") + _literal_assignment(source, "MASK_DURATION")) * 1000) == expected["nominal_trial_ms"],
    }
    probe_path = sequence_dir / "probe_positions_formal.csv"
    probes = pd.read_csv(probe_path, encoding="utf-8-sig")
    by_block = {
        int(block): sorted(group["probe_after_trial"].astype(int).tolist())
        for block, group in probes.groupby("block_num")
    }
    checks["probe_positions"] = by_block == {
        int(block): sorted(positions) for block, positions in expected["probe_positions"].items()
    }
    checks["probe_count"] = len(probes) == sum(len(positions) for positions in expected["probe_positions"].values())
    checks["schedule_version"] = probes["schedule_version"].drop_duplicates().tolist() == [expected["schedule_version"]]
    sequence_rows = {}
    sequence_sources = []
    expected_rows = expected["trials_per_cycle"] * expected["cycles_per_block"]
    for block_num, condition in enumerate(expected["block_order"], start=1):
        block_specific = sequence_dir / f"formal_{condition}{block_num}.csv"
        legacy_shared = sequence_dir / f"formal_{condition}.csv"
        sequence_path = block_specific if block_specific.exists() else legacy_shared
        frame = pd.read_csv(sequence_path)
        key = f"{condition}{block_num}" if block_specific.exists() else condition
        sequence_rows[key] = len(frame)
        checks[f"formal_{key}_rows"] = len(frame) == expected_rows
        sequence_sources.append(sequence_path)
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "sequence_rows": sequence_rows,
        "sources": [sart, probe_path, *sequence_sources],
    }



