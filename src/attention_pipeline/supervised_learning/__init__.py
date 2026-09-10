"""FocusWave supervised-learning core for the current Q1 prediction line.

This package is the production path for the 1.15.7 Task A redesign.  Historical
``multimodal_formal`` code remains available for provenance/reproduction, but new
Q1 binary development should enter through this package.
"""
from __future__ import annotations

SUPERVISED_LEARNING_VERSION = "supervised-learning-v1.0.0-dev"

from .task import Q1_BINARY_SPEC, BinaryTaskSpec, encode_q1_binary, positive_class_probability

__all__ = [
    "SUPERVISED_LEARNING_VERSION",
    "BinaryTaskSpec",
    "Q1_BINARY_SPEC",
    "encode_q1_binary",
    "positive_class_probability",
]
