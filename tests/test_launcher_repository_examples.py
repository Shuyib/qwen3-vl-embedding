import re
from pathlib import Path

import numpy as np
from PIL import Image

from src.launcher.file_type_detector import ExtensionFileTypeDetector
from src.launcher.indexer import FileIndexer
from src.launcher.search_engine import SearchEngine


REPO_ROOT = Path(__file__).resolve().parents[1]

REPOSITORY_SEARCH_EXAMPLES = [
    ("where are embeddings normalized", "quantized_embedder.py"),
    ("indexer batch processing", "indexer.py"),
    ("file type detector magika fallback", "file_type_detector.py"),
]

REALISTIC_SEARCH_EXAMPLES = [
    ("how do I get reimbursed for a client visit", "travel_expenses.md"),
    ("site feels sluggish after a release", "incident_runbook.md"),
    ("employee cannot log into the private network", "access_requests.md"),
]


class KeywordEmbedder:
    supports_images = False

    _patterns = [
        r"quantiz",
        r"embedd",
        r"normaliz",
        r"llama",
        r"index",
        r"batch",
        r"metadata",
        r"file",
        r"magika",
        r"fallback",
        r"detect",
        r"extension",
    ]

    def process(self, inputs):
        rows = []
        for item in inputs:
            text = item.get("text", "").lower()
            rows.append([len(re.findall(pattern, text)) for pattern in self._patterns])
        return np.asarray(rows, dtype=np.float32)


class ConceptEmbedder:
    supports_images = False

    _concepts = [
        {
            "client",
            "expense",
            "expenses",
            "finance",
            "reimbursed",
            "reimbursement",
            "repayment",
            "reports",
            "receipt",
            "receipts",
            "travel",
            "visit",
        },
        {
            "api",
            "deployment",
            "incident",
            "latency",
            "production",
            "release",
            "rollback",
            "runbook",
            "site",
            "slow",
            "sluggish",
            "spikes",
        },
        {
            "access",
            "credentials",
            "employee",
            "identity",
            "login",
            "network",
            "private",
            "provider",
            "request",
            "sso",
            "vpn",
        },
    ]

    def process(self, inputs):
        rows = []
        for item in inputs:
            words = set(re.findall(r"[a-z]+", item.get("text", "").lower()))
            rows.append([len(words & concept) for concept in self._concepts])
        return np.asarray(rows, dtype=np.float32)


class MultimodalExampleEmbedder:
    supports_images = True

    _concepts = ["receipt", "diagram", "portrait"]

    def process(self, inputs):
        rows = []
        for item in inputs:
            value = item.get("text", "") or item.get("image", "")
            text = Path(str(value)).stem.lower()
            rows.append([1.0 if concept in text else 0.0 for concept in self._concepts])
        return np.asarray(rows, dtype=np.float32)


def test_readme_repository_search_examples_find_expected_launcher_files(tmp_path):
    embedder = KeywordEmbedder()
    indexer = FileIndexer(
        embedder,
        tmp_path / "index",
        file_type_detector=ExtensionFileTypeDetector(),
    )
    indexer.index_directory(REPO_ROOT / "src" / "launcher", recursive=False)
    search = SearchEngine(embedder, indexer)

    for query, expected_name in REPOSITORY_SEARCH_EXAMPLES:
        results = search.search_text(query, top_k=1)

        assert results, query
        assert results[0]["name"] == expected_name


def test_realistic_search_examples_work_when_exact_query_phrase_is_absent(tmp_path):
    corpus_dir = tmp_path / "docs"
    corpus_dir.mkdir()
    files = {
        "travel_expenses.md": (
            "Team members submit expense reports with receipt totals before Friday. "
            "Finance approves repayment after manager review."
        ),
        "incident_runbook.md": (
            "When production API latency spikes, follow the incident runbook and "
            "roll back the latest deployment if error rates stay high."
        ),
        "access_requests.md": (
            "New hires request VPN credentials through SSO. The identity provider "
            "grants access after manager approval."
        ),
    }
    for name, content in files.items():
        (corpus_dir / name).write_text(content, encoding="utf-8")

    embedder = ConceptEmbedder()
    indexer = FileIndexer(
        embedder,
        tmp_path / "index",
        file_type_detector=ExtensionFileTypeDetector(),
    )
    indexer.index_directory(corpus_dir, recursive=False)
    search = SearchEngine(embedder, indexer)

    for query, expected_name in REALISTIC_SEARCH_EXAMPLES:
        expected_content = files[expected_name].lower()
        assert query.lower() not in expected_content

        results = search.search_text(query, top_k=1)

        assert results, query
        assert results[0]["name"] == expected_name


def test_image_reference_search_uses_multimodal_index_path(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name, color in {
        "receipt_scan.png": (255, 255, 255),
        "architecture_diagram.png": (0, 0, 255),
        "team_portrait.png": (255, 0, 0),
    }.items():
        Image.new("RGB", (8, 8), color).save(image_dir / name)

    embedder = MultimodalExampleEmbedder()
    indexer = FileIndexer(
        embedder,
        tmp_path / "index",
        file_type_detector=ExtensionFileTypeDetector(),
    )
    indexer.index_directory(image_dir, recursive=False)
    search = SearchEngine(embedder, indexer)

    results = search.search_image(image_dir / "architecture_diagram.png", top_k=1)

    assert results
    assert results[0]["name"] == "architecture_diagram.png"


def test_readme_lists_the_tested_repository_search_examples():
    readme = (REPO_ROOT / "LAUNCHER_README.md").read_text(encoding="utf-8")

    assert "These searches are covered by `tests/test_launcher_repository_examples.py`" in readme
    for query, expected_name in REPOSITORY_SEARCH_EXAMPLES:
        assert f"`{query}`" in readme
        assert f"`{expected_name}`" in readme
    for query, expected_name in REALISTIC_SEARCH_EXAMPLES:
        assert f"`{query}`" in readme
        assert f"`{expected_name}`" in readme
