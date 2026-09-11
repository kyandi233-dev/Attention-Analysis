"""FocusWave supervised-learning core for the current Q1-report prediction line.

This package is the production path for the 1.16.1 participant-equal redesign.
Historical ``multimodal_formal`` code remains available for provenance and
reproduction, but new first-round Q1 supervised development should enter through
this package and the frozen feature-registry interface.
"""
from __future__ import annotations

SUPERVISED_LEARNING_VERSION = "supervised-learning-v1.1.0-dev"

from .task import Q1_BINARY_SPEC, BinaryTaskSpec, encode_q1_binary, positive_class_probability

__all__ = [
    "SUPERVISED_LEARNING_VERSION",
    "BinaryTaskSpec",
    "Q1_BINARY_SPEC",
    "encode_q1_binary",
    "positive_class_probability",
]
