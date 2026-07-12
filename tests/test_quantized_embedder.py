import numpy as np

from src.launcher.quantized_embedder import QuantizedEmbedder


class StubLlamaModel:
    def create_embedding(self, text):
        embeddings = {
            "short": [[1.0, 3.0], [3.0, 5.0]],
            "longer": [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        }
        return {"data": [{"embedding": embeddings[text]}]}


def test_quantized_embedder_pools_token_embeddings_to_fixed_width_vectors():
    embedder = object.__new__(QuantizedEmbedder)
    embedder.model = StubLlamaModel()
    embedder.embed_dim = 2

    embeddings = embedder.process([{"text": "short"}, {"text": "longer"}])

    assert embeddings.shape == (2, 2)
    np.testing.assert_allclose(embeddings[0], [2.0, 4.0])
    np.testing.assert_allclose(embeddings[1], [4.0, 6.0])
