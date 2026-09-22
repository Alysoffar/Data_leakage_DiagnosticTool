"""Canonical project paths used by the detector and data preparation tools."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve the repository's shared data and generated-output locations."""

    root: Path

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ProjectPaths":
        """Build paths from a file located below the repository's ``src`` directory."""
        return cls(Path(file_path).resolve().parents[2])

    @property
    def data(self) -> Path:
        return self.root / "Data"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def raw_data(self) -> Path:
        return self.data / "raw"

    @property
    def processed_data(self) -> Path:
        return self.data / "processed"

    @property
    def clean_data(self) -> Path:
        return self.processed_data / "clean"

    @property
    def leaky_data(self) -> Path:
        return self.processed_data / "leaky"

    def dataset_data(self, dataset_name: str | None = None) -> Path:
        """Return the processed root for the default or a named dataset."""
        return self.processed_data if dataset_name is None else self.processed_data / dataset_name

    def report_path(self, filename: str = "leak_report.md") -> Path:
        """Return a report path inside the canonical reports directory."""
        return self.reports / filename

    def ensure_output_directories(self) -> None:
        """Create directories used for generated artifacts."""
        self.reports.mkdir(parents=True, exist_ok=True)
        self.raw_data.mkdir(parents=True, exist_ok=True)
        self.clean_data.mkdir(parents=True, exist_ok=True)
        self.leaky_data.mkdir(parents=True, exist_ok=True)
