"""Validate the frozen supervised feature registry before any real Task-B/Task-A run.

This gate answers two independent questions. Neither may be relaxed, and the second was
previously missing entirely:

1. **Producer-side time legality** (``time_legality_audit``): every feature that requests
   any formal prediction eligibility proves ``verified_pre_probe_only`` with non-blank
   evidence, so the runtime cannot be handed a leakage-unsafe predictor.
2. **Registry structural consistency** (new): the frozen entries can actually be turned
   into an executable comparison plan. Before this was added, the time-legality audit
   alone would happily pass a registry whose entries collide (two representations of one
   scientific feature in the full model, a device package referencing a device the
   feature does not require, a duplicated predictor column, and so on) and which could
   therefore never produce a legitimate model.

The structural half is strictly additive: it never weakens a judgement, it reuses the
same contract functions the real run uses, and every pre-existing output key is preserved
verbatim so existing consumers keep working.

Failures propagate as contract errors, so the process exits non-zero and no downstream
step can mistake a rejected registry for an accepted one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.feature_registry import (
    build_feature_comparison_plan,
    load_registered_features,
)
from attention_pipeline.supervised_learning.time_legality import time_legality_audit


def _structure_and_plan(section: object) -> tuple[dict[str, object], dict[str, object]]:
    """Validate registry structure and summarise the generated comparison plan.

    Parameters
    ----------
    section : object
        The ``feature_registry`` mapping read from the scientific config.

    Returns
    -------
    (registry_structure, comparison_plan) : tuple of JSON-serialisable dicts
        ``registry_structure`` describes the frozen entries; ``comparison_plan`` is the
        full auditable model plan the registry generates.
    """
    if not isinstance(section, dict):
        raise TypeError("feature_registry must be a mapping")
    entries = section.get("features", [])
    if not entries:
        return (
            {
                "status": "empty_registry",
                "n_features": 0,
                "feature_ids": [],
                "note": "an empty registry generates no model plan; real runs fail closed",
            },
            {"status": "empty_registry", "model_ids": [], "models": []},
        )

    registry = load_registered_features(section)
    plan = build_feature_comparison_plan(registry)
    audit = plan.audit_dict()
    models = audit["models"]

    structure = {
        "status": "ok",
        "n_features": len(registry),
        "feature_ids": [feature.feature_id for feature in registry],
        "scientific_modalities": sorted({feature.modality for feature in registry}),
        "source_namespaces": sorted({feature.source_namespace for feature in registry}),
        "scientific_feature_ids": sorted(
            {feature.scientific_feature_id for feature in registry}
        ),
        "n_models": len(models),
        "unavailable_modalities": audit["unavailable_modalities"],
    }
    comparison_plan = {
        "status": "ok",
        "model_ids": [model["model_id"] for model in models],
        "models": models,
        "behavior_increment_pairs": audit["behavior_increment_pairs"],
        "full_leave_one_out_pairs": audit["full_leave_one_out_pairs"],
        "modality_increment_pairs": audit["modality_increment_pairs"],
        "full_leave_one_modality_out_pairs": audit["full_leave_one_modality_out_pairs"],
        "device_package_model_ids": audit["device_package_model_ids"],
        "unavailable_device_packages": audit["unavailable_device_packages"],
    }
    return structure, comparison_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the FocusWave frozen feature registry time-legality and structural "
            "contracts before any real Task-B/Task-A supervised run."
        )
    )
    parser.add_argument("--config", default="configs/supervised_learning_v1.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON audit output path")
    args = parser.parse_args()

    config = load_config(args.config, paths_config=args.paths_config)
    section = config.data.get("feature_registry", {})

    audit = time_legality_audit(section)
    structure, comparison_plan = _structure_and_plan(section)
    audit["registry_structure"] = structure
    audit["comparison_plan"] = comparison_plan

    payload = json.dumps(audit, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output}")

    print(
        json.dumps(
            {
                "registry_validation_summary": {
                    "schema_version": audit.get("schema_version"),
                    "config_digest": config.digest,
                    "n_features": audit.get("n_features"),
                    "n_verified_pre_probe_only": audit.get("n_verified_pre_probe_only"),
                    "pending_or_blocked_feature_ids": audit.get("pending_or_blocked_feature_ids"),
                    "registry_structure_status": structure.get("status"),
                    "n_models": structure.get("n_models"),
                    "unavailable_modalities": structure.get("unavailable_modalities"),
                    "unavailable_device_packages": comparison_plan.get(
                        "unavailable_device_packages"
                    ),
                    "device_package_model_ids": comparison_plan.get("device_package_model_ids"),
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
