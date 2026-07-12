import numpy as np

from src.launcher.file_type_detector import FileTypeInfo
from src.launcher.indexer import FileIndexer


class StubEmbedder:
    def __init__(self):
        self.inputs = []

    def process(self, inputs):
        self.inputs.extend(inputs)
        return np.ones((len(inputs), 3), dtype=np.float32)


class TextOnlyStubEmbedder(StubEmbedder):
    supports_images = False


class StubDetector:
    def __init__(self, results):
        self.results = results

    def detect(self, file_path):
        return self.results[file_path.name]


def test_indexer_indexes_magika_detected_extensionless_text(tmp_path):
    document = tmp_path / "Dockerfile"
    document.write_text("FROM python:3.12\n", encoding="utf-8")
    ignored = tmp_path / "archive.bin"
    ignored.write_bytes(b"\x00\x01\x02")

    embedder = StubEmbedder()
    detector = StubDetector(
        {
            "Dockerfile": FileTypeInfo(
                label="dockerfile",
                description="Dockerfile",
                mime_type="text/x-dockerfile",
                group="code",
                is_text=True,
                launcher_type="text",
                score=0.99,
                source="magika",
            ),
            "archive.bin": FileTypeInfo(
                label="unknown",
                description="Unknown binary data",
                mime_type="application/octet-stream",
                group="unknown",
                is_text=False,
                launcher_type="unknown",
                source="magika",
            ),
        }
    )
    indexer = FileIndexer(embedder, tmp_path / "index", file_type_detector=detector)

    indexer.index_directory(tmp_path, recursive=False)

    assert [item["name"] for item in indexer.file_metadata] == ["Dockerfile"]
    metadata = indexer.file_metadata[0]
    assert metadata["type"] == "text"
    assert metadata["file_type"]["label"] == "dockerfile"
    assert metadata["file_type"]["source"] == "magika"
    assert embedder.inputs[0]["text"].startswith("File name: Dockerfile\nFile type: Dockerfile")


def test_indexer_skips_unsupported_detected_files(tmp_path):
    unsupported = tmp_path / "payload"
    unsupported.write_bytes(b"\x00\x01\x02")

    embedder = StubEmbedder()
    detector = StubDetector(
        {
            "payload": FileTypeInfo(
                label="unknown",
                description="Unknown binary data",
                mime_type="application/octet-stream",
                group="unknown",
                is_text=False,
                launcher_type="unknown",
                source="magika",
            )
        }
    )
    indexer = FileIndexer(embedder, tmp_path / "index", file_type_detector=detector)

    indexer.index_directory(tmp_path, recursive=False)

    assert indexer.file_metadata == []
    assert embedder.inputs == []


def test_indexer_skips_images_for_text_only_embedder(tmp_path):
    document = tmp_path / "notes.md"
    document.write_text("Launcher notes\n", encoding="utf-8")
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fake image bytes")

    embedder = TextOnlyStubEmbedder()
    detector = StubDetector(
        {
            "notes.md": FileTypeInfo(
                label="md",
                description="Markdown text file",
                mime_type="text/markdown",
                group="text",
                is_text=True,
                launcher_type="text",
                source="test",
            ),
            "diagram.png": FileTypeInfo(
                label="png",
                description="PNG image file",
                mime_type="image/png",
                group="image",
                is_text=False,
                launcher_type="image",
                source="test",
            ),
        }
    )
    indexer = FileIndexer(embedder, tmp_path / "index", file_type_detector=detector)

    indexer.index_directory(tmp_path, recursive=False)

    assert [item["name"] for item in indexer.file_metadata] == ["notes.md"]
    assert len(embedder.inputs) == 1
    assert "text" in embedder.inputs[0]
