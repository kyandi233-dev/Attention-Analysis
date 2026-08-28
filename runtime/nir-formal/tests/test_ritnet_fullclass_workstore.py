from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ritnet_fullclass_workstore import (
    FullClassWorkStore,
    WORKSTORE_IDENTITY_MISMATCH_REASON,
    archive_identity_mismatch_workstore,
)


def test_identity_mismatch_archives_sqlite_and_sidecars_without_reuse(tmp_path: Path) -> None:
    work_root = tmp_path / ".ritnet-fullclass-work"
    path = work_root / "sub-056.sqlite"
    with FullClassWorkStore(path, identity={"run": "old"}):
        pass
    Path(str(path) + "-wal").write_bytes(b"old-wal")
    Path(str(path) + "-shm").write_bytes(b"old-shm")

    archived = archive_identity_mismatch_workstore(path)

    assert archived is not None
    assert not path.exists()
    assert not Path(str(path) + "-wal").exists()
    assert not Path(str(path) + "-shm").exists()
    record = json.loads((archived / "_archive_reason.json").read_text(encoding="utf-8"))
    assert record["reason"] == WORKSTORE_IDENTITY_MISMATCH_REASON
    assert (archived / "sub-056.sqlite").is_file()
    assert (archived / "sub-056.sqlite-wal").is_file()
    assert (archived / "sub-056.sqlite-shm").is_file()

    with FullClassWorkStore(path, identity={"run": "new"}) as store:
        assert store.stored_rows == 0
