"""Fetch the pinned upstream RITnet weights needed only to regenerate ONNX files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Callable


PACKAGE_ROOT = Path(__file__).resolve().parent
OFFICIAL_UPSTREAM_COMMIT = "6431c57ce7bf0eda935fb6178b926ae9440b50bf"
OFFICIAL_WEIGHTS_GIT_BLOB_SHA1 = "f0864e6651f578525a9101c7ca787e23d2d201d7"
OFFICIAL_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/AayushKrChaudhary/RITnet/"
    f"{OFFICIAL_UPSTREAM_COMMIT}/best_model.pkl"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "models" / "ritnet-best_model.pkl"


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def verify_weights_bytes(payload: bytes) -> str:
    digest = git_blob_sha1(payload)
    if digest != OFFICIAL_WEIGHTS_GIT_BLOB_SHA1:
        raise RuntimeError(
            "Downloaded RITnet best_model.pkl does not match the pinned upstream Git blob: "
            f"{digest} != {OFFICIAL_WEIGHTS_GIT_BLOB_SHA1}"
        )
    return digest


def fetch_weights(
    output: Path,
    *,
    url: str = OFFICIAL_WEIGHTS_URL,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> dict[str, object]:
    output = Path(output).expanduser().resolve()
    if output.is_file():
        payload = output.read_bytes()
        digest = verify_weights_bytes(payload)
        return {
            "status": "existing_valid",
            "path": str(output),
            "size_bytes": len(payload),
            "git_blob_sha1": digest,
            "upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with opener(url, timeout=60) as response:
        payload = response.read()
    digest = verify_weights_bytes(payload)

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "status": "downloaded_verified",
        "path": str(output),
        "size_bytes": len(payload),
        "git_blob_sha1": digest,
        "upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
        "source_url": url,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the pinned upstream RITnet best_model.pkl for ONNX regeneration"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = fetch_weights(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
