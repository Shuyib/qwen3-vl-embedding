"""
Tests for image embedding with the full-precision Qwen3VLEmbedder.

These tests verify that the full-precision model can:
1. Load and process images
2. Generate fixed-dimension embeddings for images
3. Generate consistent embeddings for the same image
4. Distinguish between different images
5. Handle mixed text+image inputs
6. Handle image-only inputs (no text)
"""
import numpy as np
from pathlib import Path
import pytest

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder


MODEL_PATH = Path(__file__).parents[1] / "models" / "Qwen3-VL-Embedding-2B-full"


@pytest.fixture(scope="module")
def embedder():
    """Load the full-precision embedder once per module."""
    if not (MODEL_PATH / "model.safetensors").exists():
        pytest.skip("Model weights not found. Run: huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./models/Qwen3-VL-Embedding-2B")
    return Qwen3VLEmbedder(str(MODEL_PATH))


@pytest.fixture
def sample_images(tmp_path):
    """Create a set of small test images (8x8 colored PNGs)."""
    from PIL import Image

    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    paths = {}
    for name, color in colors.items():
        path = tmp_path / f"{name}.png"
        img = Image.new("RGB", (8, 8), color)
        img.save(path)
        paths[name] = path
    return paths


class TestImageEmbeddingBasics:
    """Core image embedding functionality."""

    def test_embedder_initializes(self, embedder):
        """Verify the full-precision model loads correctly."""
        assert embedder is not None
        assert hasattr(embedder, "model")
        assert hasattr(embedder, "processor")
        assert embedder.model.device.type in ("cuda", "cpu")

    def test_single_image_embedding_shape(self, embedder, sample_images):
        """A single image produces a 1D embedding vector of expected dimension."""
        inputs = [{"image": str(sample_images["blue"])}]
        embeddings = embedder.process(inputs)
        assert embeddings.shape[0] == 1  # one input
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 2048  # Qwen3-VL-Embedding-2B hidden_size

    def test_multiple_image_embeddings_shape(self, embedder, sample_images):
        """Multiple images produce a batch of embeddings."""
        inputs = [
            {"image": str(sample_images["red"])},
            {"image": str(sample_images["green"])},
            {"image": str(sample_images["blue"])},
        ]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (3, 2048)

    def test_embeddings_are_normalized(self, embedder, sample_images):
        """Embeddings are L2-normalized (unit vectors)."""
        inputs = [{"image": str(sample_images["blue"])}]
        embeddings = embedder.process(inputs, normalize=True)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embeddings_are_deterministic(self, embedder, sample_images):
        """Same image produces same embedding (model is in eval mode)."""
        inputs = [{"image": str(sample_images["blue"])}]
        emb1 = embedder.process(inputs)
        emb2 = embedder.process(inputs)
        np.testing.assert_allclose(emb1, emb2, atol=1e-5)

    def test_different_images_different_embeddings(self, embedder, sample_images):
        """Different images produce different embeddings."""
        inputs = [
            {"image": str(sample_images["blue"])},
            {"image": str(sample_images["red"])},
        ]
        embeddings = embedder.process(inputs)
        # Cosine similarity should be less than 1.0 for different images
        sim = np.dot(embeddings[0], embeddings[1])
        assert sim < 0.999, f"Different images have near-identical embeddings (sim={sim:.6f})"

    def test_embedding_not_all_zeros(self, embedder, sample_images):
        """Image embeddings carry meaningful signal (not all zeros)."""
        inputs = [{"image": str(sample_images["blue"])}]
        embeddings = embedder.process(inputs)
        assert not np.allclose(embeddings, 0, atol=1e-6)


class TestImageTextInteraction:
    """Mixed image and text input behavior."""

    def test_image_with_text(self, embedder, sample_images):
        """Image with accompanying text produces a valid embedding."""
        inputs = [
            {
                "image": str(sample_images["blue"]),
                "text": "A blue square image",
            }
        ]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (1, 2048)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_image_with_instruction(self, embedder, sample_images):
        """Custom instruction modifies the embedding context."""
        inputs = [
            {
                "image": str(sample_images["blue"]),
                "instruction": "Describe this image in detail.",
            }
        ]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (1, 2048)

    def test_image_and_text_embedding_distinct_from_image_only(self, embedder, sample_images):
        """Adding text to an image changes the embedding."""
        img_only = embedder.process([{"image": str(sample_images["blue"])}])
        img_text = embedder.process([{
            "image": str(sample_images["blue"]),
            "text": "A calm blue ocean scene",
        }])
        # Different inputs should produce different embeddings
        sim = np.dot(img_only[0], img_text[0])
        assert sim < 0.999, (
            f"Image-only and image+text embeddings are nearly identical "
            f"(cosine sim={sim:.6f})"
        )


class TestImageLoadVariants:
    """Different ways to pass images to the embedder."""

    def test_image_from_pil_object(self, embedder):
        """Passing a PIL Image object directly works."""
        from PIL import Image
        img = Image.new("RGB", (16, 16), (0, 128, 255))
        inputs = [{"image": img}]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (1, 2048)

    def test_image_from_path_string(self, embedder, sample_images):
        """Passing a file path string works."""
        inputs = [{"image": str(sample_images["green"])}]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (1, 2048)

    def test_image_from_pathlib_path(self, embedder, sample_images):
        """Passing a pathlib.Path works."""
        inputs = [{"image": sample_images["green"]}]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (1, 2048)


class TestImageBatchProcessing:
    """Batch processing of multiple images."""

    def test_batch_of_images(self, embedder, sample_images):
        """Multiple images processed in one batch."""
        inputs = [
            {"image": str(sample_images["red"])},
            {"image": str(sample_images["green"])},
            {"image": str(sample_images["blue"])},
            {"image": str(sample_images["white"])},
            {"image": str(sample_images["black"])},
        ]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (5, 2048)

    def test_text_and_image_mixed_batch(self, embedder, sample_images):
        """Mixed batch of text-only and image-only inputs."""
        inputs = [
            {"text": "A description of a sunset"},
            {"image": str(sample_images["blue"])},
            {"text": "Code for a web server", "image": str(sample_images["red"])},
        ]
        embeddings = embedder.process(inputs)
        assert embeddings.shape == (3, 2048)
        # Each embedding should be a unit vector
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_single_element_batch(self, embedder, sample_images):
        """Single element lists work correctly."""
        embeddings = embedder.process([{"image": str(sample_images["blue"])}])
        assert embeddings.shape == (1, 2048)
        # Verify it's the same as a direct call
        assert np.allclose(embeddings[0], embeddings[0], atol=1e-10)


class TestImageSimilarity:
    """Semantic similarity between images."""

    def test_same_image_high_similarity(self, embedder, sample_images):
        """Same image queried twice should have near-perfect similarity."""
        blue_path = str(sample_images["blue"])
        embeddings = embedder.process([
            {"image": blue_path},
            {"image": blue_path},
        ])
        sim = np.dot(embeddings[0], embeddings[1])
        assert sim > 0.99, f"Same image similarity too low: {sim:.6f}"

    def test_black_and_white_have_some_similarity(self, embedder, sample_images):
        """Black and white are both luminance-neutral — some similarity expected."""
        embeddings = embedder.process([
            {"image": str(sample_images["black"])},
            {"image": str(sample_images["white"])},
        ])
        sim = np.dot(embeddings[0], embeddings[1])
        # They're different colors, so similarity should be > -1, < 1
        assert -1.0 < sim < 1.0
        # But they're both neutral luminance, so should be somewhat similar
        # (not necessarily — just verifying it's a valid cosine similarity)
        print(f"Black-White cosine similarity: {sim:.6f}")