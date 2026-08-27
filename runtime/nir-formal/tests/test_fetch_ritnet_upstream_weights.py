from __future__ import annotations

import io

import pytest

import fetch_ritnet_upstream_weights as fetcher


def _git_blob_sha1_reference(payload: bytes) -> str:
    import hashlib

    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def test_git_blob_sha1_matches_git_object_definition():
    payload = b"ritnet-test-payload"
    assert fetcher.git_blob_sha1(payload) == _git_blob_sha1_reference(payload)


def test_existing_valid_weight_is_reused_without_network(monkeypatch, tmp_path):
    payload = b"known-weight"
    digest = fetcher.git_blob_sha1(payload)
    monkeypatch.setattr(fetcher, "OFFICIAL_WEIGHTS_GIT_BLOB_SHA1", digest)
    output = tmp_path / "best_model.pkl"
    output.write_bytes(payload)

    def no_network(*args, **kwargs):
        raise AssertionError("network must not be used for an already verified local weight")

    result = fetcher.fetch_weights(output, opener=no_network)
    assert result["status"] == "existing_valid"
    assert result["git_blob_sha1"] == digest
    assert output.read_bytes() == payload


def test_download_is_verified_before_publish(monkeypatch, tmp_path):
    payload = b"downloaded-weight"
    digest = fetcher.git_blob_sha1(payload)
    monkeypatch.setattr(fetcher, "OFFICIAL_WEIGHTS_GIT_BLOB_SHA1", digest)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    output = tmp_path / "best_model.pkl"
    result = fetcher.fetch_weights(output, opener=lambda *args, **kwargs: Response(payload))
    assert result["status"] == "downloaded_verified"
    assert output.read_bytes() == payload


def test_wrong_download_is_rejected_without_publishing(monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher, "OFFICIAL_WEIGHTS_GIT_BLOB_SHA1", "0" * 40)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    output = tmp_path / "best_model.pkl"
    with pytest.raises(RuntimeError, match="does not match"):
        fetcher.fetch_weights(output, opener=lambda *args, **kwargs: Response(b"wrong"))
    assert not output.exists()
