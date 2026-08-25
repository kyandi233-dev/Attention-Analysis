"""RGB behavior-analysis development module.

Current design (rgb-dev):

- Face: benchmark Py-Feat vs LibreFace 2.0 before freezing a backend.
- Pose: MediaPipe Pose as the default first-line body landmark extractor.
- Motion: OpenCV frame-difference Motion Energy as the default first-line
  movement-magnitude extractor.
- Timing: align continuous RGB outputs with FocusWave ``formaltest`` artifacts
  through RGB frame Unix timestamps, ``master_timeline.csv`` and behavior
  absolute onset times. Do not infer formal phase timing from nominal FPS.

This package is intentionally only a development skeleton at this stage. Model
adapters, QC thresholds, sampling rates and production runtime behavior are not
frozen yet.
"""

__all__: list[str] = []
