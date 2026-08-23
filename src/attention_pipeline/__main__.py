"""Current module entrypoint for Attention-Analysis.

The legacy generic CLI is retained in ``attention_pipeline.cli`` for historical
compatibility, but it is not the current formal analysis entrypoint.
"""

from __future__ import annotations


def main() -> int:
    print(
        "Attention-Analysis current entrypoints:\n"
        "  Behavior (FocusWave v3.1.3 BB):\n"
        "    python scripts/sart_formal_analysis.py --help\n"
        "  NIR formal NVIDIA/CUDA runtime:\n"
        "    cd runtime/nir-formal && python run_pipeline.py --help\n"
        "  Historical BBB v3.0 reproduction:\n"
        "    python scripts/sart_bbb_v3_0_analysis.py --help\n"
        "\n"
        "Legacy attention_pipeline.cli is retained for historical compatibility "
        "and is not the current formal pipeline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
