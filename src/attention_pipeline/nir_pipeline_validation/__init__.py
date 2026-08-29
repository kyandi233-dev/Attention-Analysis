"""Validation-only downstream analyses for completed NIR analysis tables.

The package-level ``run_validation`` symbol is the authoritative pupil-only
validation API.  The historical PIR-oriented runner remains importable only
through the explicit ``legacy_pir_run_validation`` compatibility name.
"""

from .pupil_validation import run_validation
from .run import run_validation as legacy_pir_run_validation

__all__ = ["run_validation", "legacy_pir_run_validation"]
