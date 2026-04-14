"""
File type detection helpers for the launcher indexer.

Magika is optional. When it is not installed or cannot classify a file with
enough confidence, the detector falls back to the launcher's extension map.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".css",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}

MAGIKA_GENERIC_LABELS = {"txt", "unknown", "empty", "inode/symlink"}


@dataclass(frozen=True)
class FileTypeInfo:
    """Normalized file type result used by the launcher."""

    label: str
    description: str
    mime_type: str
    group: str
    is_text: bool
    launcher_type: str
    score: Optional[float] = None
    source: str = "extension"

    @property
    def is_supported(self) -> bool:
        return self.launcher_type in {"text", "image"}

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "label": self.label,
            "description": self.description,
            "mime_type": self.mime_type,
            "group": self.group,
            "is_text": self.is_text,
            "source": self.source,
        }
        if self.score is not None:
            metadata["score"] = self.score
        return metadata


class ExtensionFileTypeDetector:
    """File type detector based on the launcher's extension allow-list."""

    source = "extension"

    def detect(self, file_path: Path) -> FileTypeInfo:
        suffix = file_path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            label = suffix[1:] or "text"
            return FileTypeInfo(
                label=label,
                description=f"{label.upper()} text file" if label else "Text file",
                mime_type="text/plain",
                group="text",
                is_text=True,
                launcher_type="text",
                source=self.source,
            )

        if suffix in IMAGE_EXTENSIONS:
            label = suffix[1:] or "image"
            mime_type = "image/svg+xml" if suffix == ".svg" else f"image/{label}"
            return FileTypeInfo(
                label=label,
                description=f"{label.upper()} image file",
                mime_type=mime_type,
                group="image",
                is_text=False,
                launcher_type="image",
                source=self.source,
            )

        return FileTypeInfo(
            label="unknown",
            description="Unknown file type",
            mime_type="application/octet-stream",
            group="unknown",
            is_text=False,
            launcher_type="unknown",
            source=self.source,
        )


class MagikaFileTypeDetector:
    """File type detector backed by Google's Magika package."""

    source = "magika"

    def __init__(self):
        try:
            from magika import Magika
        except ImportError as exc:
            raise RuntimeError("Magika is not installed. Install with: uv pip install magika") from exc

        self._magika = Magika()

    def detect(self, file_path: Path) -> FileTypeInfo:
        result = self._magika.identify_path(file_path)
        value = result.output

        launcher_type = _launcher_type_from_magika(value)
        return FileTypeInfo(
            label=str(value.label),
            description=str(value.description),
            mime_type=str(value.mime_type),
            group=str(value.group),
            is_text=bool(value.is_text),
            launcher_type=launcher_type,
            score=float(result.score) if result.score is not None else None,
            source=self.source,
        )


class AutoFileTypeDetector:
    """Use Magika when available and fall back to extension detection."""

    def __init__(self, prefer_magika: bool = True):
        self._extension_detector = ExtensionFileTypeDetector()
        self._magika_detector: Optional[MagikaFileTypeDetector] = None

        if prefer_magika:
            try:
                self._magika_detector = MagikaFileTypeDetector()
                logger.info("Using Magika for file type detection")
            except RuntimeError as exc:
                logger.info("%s; falling back to extension-based detection", exc)

    def detect(self, file_path: Path) -> FileTypeInfo:
        extension_info = self._extension_detector.detect(file_path)

        if self._magika_detector is None:
            return extension_info

        try:
            magika_info = self._magika_detector.detect(file_path)
        except Exception as exc:
            logger.warning("Magika failed to identify %s: %s", file_path, exc)
            return extension_info

        if magika_info.is_supported:
            return magika_info

        if extension_info.is_supported:
            return extension_info

        return magika_info


def make_file_type_detector(mode: str):
    if mode == "extension":
        return ExtensionFileTypeDetector()
    if mode == "magika":
        return MagikaFileTypeDetector()
    if mode == "auto":
        return AutoFileTypeDetector()
    raise ValueError(f"Unknown file type detector mode: {mode}")


def _launcher_type_from_magika(value: Any) -> str:
    group = str(value.group)
    label = str(value.label)
    mime_type = str(value.mime_type)

    if group == "image" or mime_type.startswith("image/"):
        return "image"

    if bool(value.is_text) and label not in MAGIKA_GENERIC_LABELS:
        return "text"

    if bool(value.is_text) and group in {"text", "code"}:
        return "text"

    return "unknown"
