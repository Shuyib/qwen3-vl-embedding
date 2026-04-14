from types import SimpleNamespace

from src.launcher.file_type_detector import AutoFileTypeDetector, ExtensionFileTypeDetector, FileTypeInfo


def test_extension_detector_identifies_known_text_file(tmp_path):
    path = tmp_path / "config.py"
    path.write_text("print('hello')", encoding="utf-8")

    result = ExtensionFileTypeDetector().detect(path)

    assert result.is_supported
    assert result.launcher_type == "text"
    assert result.label == "py"
    assert result.is_text is True


def test_extension_detector_rejects_unknown_binary(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00\x01\x02")

    result = ExtensionFileTypeDetector().detect(path)

    assert not result.is_supported
    assert result.launcher_type == "unknown"


def test_auto_detector_uses_magika_result_for_extensionless_text(tmp_path):
    path = tmp_path / "Dockerfile"
    path.write_text("FROM python:3.12\n", encoding="utf-8")

    detector = AutoFileTypeDetector(prefer_magika=False)
    detector._magika_detector = SimpleNamespace(
        detect=lambda _: FileTypeInfo(
            label="dockerfile",
            description="Dockerfile",
            mime_type="text/x-dockerfile",
            group="code",
            is_text=True,
            launcher_type="text",
            score=0.99,
            source="magika",
        )
    )

    result = detector.detect(path)

    assert result.launcher_type == "text"
    assert result.source == "magika"
    assert result.label == "dockerfile"


def test_auto_detector_falls_back_to_extension_when_magika_is_unknown(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n", encoding="utf-8")

    detector = AutoFileTypeDetector(prefer_magika=False)
    detector._magika_detector = SimpleNamespace(
        detect=lambda _: FileTypeInfo(
            label="unknown",
            description="Unknown binary data",
            mime_type="application/octet-stream",
            group="unknown",
            is_text=False,
            launcher_type="unknown",
            source="magika",
        )
    )

    result = detector.detect(path)

    assert result.launcher_type == "text"
    assert result.source == "extension"
    assert result.label == "md"
