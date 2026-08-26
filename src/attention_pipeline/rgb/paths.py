from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from attention_pipeline.config import Config


@dataclass(frozen=True)
class RGBOutputLayout:
    """Canonical external output layout for RGB analysis results.

    The layout is intentionally shallow:
    - dataset-level QC/summary files live directly under Beijing-RGB;
    - temporary/pilot outputs live under _test;
    - formal per-subject outputs live under sub-XXX/;
    - every per-subject filename repeats the subject id prefix so files remain
      identifiable when copied outside their original directory.
    """

    root: Path
    test_dir_name: str = "_test"

    @classmethod
    def from_config(cls, config: Config) -> "RGBOutputLayout":
        output = config.section("output")
        raw_root = Path(str(output.get("root", "D:/_AttentionData/Beijing-RGB")))
        if not raw_root.is_absolute():
            raw_root = (config.path.parent.parent / raw_root).resolve()
        return cls(root=raw_root, test_dir_name=str(output.get("test_dir", "_test")))

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def dataset_file(self, filename: str) -> Path:
        return self.ensure_root() / filename

    def test_dir(self) -> Path:
        path = self.ensure_root() / self.test_dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_file(self, filename: str) -> Path:
        return self.test_dir() / filename

    def subject_dir(self, subject: str) -> Path:
        path = self.ensure_root() / subject
        path.mkdir(parents=True, exist_ok=True)
        return path

    def subject_file(self, subject: str, suffix: str) -> Path:
        """Return sub-XXX/sub-XXX_<suffix>, creating only the subject directory."""
        clean_suffix = suffix.lstrip("_")
        return self.subject_dir(subject) / f"{subject}_{clean_suffix}"
