"""Downstream NIR × formal-SART time alignment and feature extraction.

This package reads frozen FocusWave v3.1.3 BB behavior and frozen full-class
NIR outputs. It does not modify the formal NIR runtime or raw behavior files.
"""

from .contract import ALIGNMENT_PIPELINE_VERSION, ALIGNMENT_SCHEMA_VERSION

__all__ = ["ALIGNMENT_PIPELINE_VERSION", "ALIGNMENT_SCHEMA_VERSION"]
