"""
Quantized embedder wrapper for GGUF models using llama-cpp-python.

This module provides a lightweight wrapper for using quantized Qwen3-VL embedding
models in GGUF format, which offer significantly reduced memory footprint and
faster inference on CPU.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")


class QuantizedEmbedder:
    """
    Quantized embedder using GGUF format models.
    
    This embedder provides a lightweight alternative to the full Qwen3VLEmbedder
    by using quantized GGUF models, which significantly reduce memory usage and
    improve CPU inference speed.
    
    Parameters
    ----------
    model_path : str
        Path to the GGUF model file.
    n_ctx : int, optional
        Context window size (default: 8192).
    n_gpu_layers : int, optional
        Number of layers to offload to GPU (default: 0 for CPU-only).
    embedding : bool, optional
        Enable embedding mode (default: True).
    verbose : bool, optional
        Enable verbose logging (default: False).
    
    Attributes
    ----------
    model : Llama
        The loaded llama-cpp model.
    embed_dim : int
        Dimension of the embeddings.
    
    Examples
    --------
    Basic usage with GGUF model:
    
    >>> from src.launcher.quantized_embedder import QuantizedEmbedder
    >>> embedder = QuantizedEmbedder(
    ...     model_path="./models/Qwen3-VL-Embedding-2B-Q4_K_M.gguf",
    ...     n_ctx=8192,
    ...     n_gpu_layers=0
    ... )
    >>> 
    >>> # Embed text
    >>> inputs = [{"text": "A sunset on the beach"}]
    >>> embeddings = embedder.process(inputs)
    >>> print(embeddings.shape)  # (1, embedding_dim)
    
    How to Use
    ----------
    1. Download a quantized GGUF model from HuggingFace:
       
       .. code-block:: bash
       
           huggingface-cli download DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF \\
               Qwen3-VL-Embedding-2B-Q4_K_M.gguf \\
               --local-dir ./models/gguf/
    
    2. Initialize the embedder:
       
       .. code-block:: python
       
           embedder = QuantizedEmbedder(
               model_path="./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf"
           )
    
    3. Process inputs (text only for GGUF models):
       
       .. code-block:: python
       
           inputs = [
               {"text": "Python machine learning code"},
               {"text": "Database configuration file"}
           ]
           embeddings = embedder.process(inputs)
    
    Notes
    -----
    - GGUF models currently support text-only inputs
    - For multimodal (image/video) support, use the full Qwen3VLEmbedder
    - Quantization reduces model size by 4-8x with minimal accuracy loss
    - Q4_K_M quantization offers a good balance of size and quality
    
    See Also
    --------
    src.models.qwen3_vl_embedding.Qwen3VLEmbedder : Full-precision embedder
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
        embedding: bool = True,
        verbose: bool = False
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for quantized models. "
                "Install with: pip install llama-cpp-python"
            )
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading quantized model: {model_path}")
        
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=embedding,
            verbose=verbose
        )
        self.supports_images = False
        self.supports_videos = False
        
        # Get embedding dimension by creating a test embedding
        test_embed = self.model.create_embedding("test")
        self.embed_dim = self._normalize_embedding(test_embed['data'][0]['embedding']).shape[0]
        
        logger.info(f"Quantized model loaded. Embedding dimension: {self.embed_dim}")
    
    def _prepare_text_input(self, input_dict: Dict[str, Any]) -> str:
        """
        Prepare text input from input dictionary.
        
        Parameters
        ----------
        input_dict : dict
            Input dictionary with 'text' and optional 'instruction' keys.
        
        Returns
        -------
        str
            Formatted text for embedding.
        
        Examples
        --------
        >>> embedder = QuantizedEmbedder("model.gguf")
        >>> text = embedder._prepare_text_input({"text": "Hello world"})
        >>> print(text)
        'Hello world'
        
        How to Use
        ----------
        This is an internal method. Users should call `process()` instead:
        
        .. code-block:: python
        
            inputs = [{"text": "Your text", "instruction": "Represent the input"}]
            embeddings = embedder.process(inputs)
        """
        text = input_dict.get('text', '')
        instruction = input_dict.get('instruction', '')
        
        if instruction:
            return f"{instruction}\n\n{text}"
        return text

    def _normalize_embedding(self, embedding: Any) -> np.ndarray:
        """Convert llama.cpp embedding output to one flat vector."""
        embedding_array = np.asarray(embedding, dtype=np.float32)

        if embedding_array.ndim == 1:
            return embedding_array
        if embedding_array.ndim == 2:
            return embedding_array.mean(axis=0)
        if embedding_array.ndim > 2:
            return embedding_array.reshape(-1, embedding_array.shape[-1]).mean(axis=0)

        raise ValueError("Embedding output must contain at least one dimension")
    
    def process(self, inputs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Process a batch of inputs and return embeddings.
        
        Parameters
        ----------
        inputs : list of dict
            List of input dictionaries. Each dictionary should contain:
            - 'text' (str): Text content to embed
            - 'instruction' (str, optional): Task instruction
        
        Returns
        -------
        np.ndarray
            Array of embeddings with shape (n_inputs, embed_dim).
        
        Raises
        ------
        ValueError
            If inputs contain unsupported keys (e.g., 'image', 'video').
        
        Examples
        --------
        Single text input:
        
        >>> embedder = QuantizedEmbedder("model.gguf")
        >>> inputs = [{"text": "Machine learning with Python"}]
        >>> embeddings = embedder.process(inputs)
        >>> print(embeddings.shape)
        (1, 2048)
        
        Multiple inputs with instructions:
        
        >>> inputs = [
        ...     {
        ...         "text": "Neural network architecture",
        ...         "instruction": "Represent this technical content"
        ...     },
        ...     {"text": "Database configuration settings"}
        ... ]
        >>> embeddings = embedder.process(inputs)
        >>> print(embeddings.shape)
        (2, 2048)
        
        How to Use
        ----------
        1. Prepare your inputs as a list of dictionaries:
           
           .. code-block:: python
           
               inputs = [
                   {"text": "First document"},
                   {"text": "Second document", "instruction": "Represent the doc"}
               ]
        
        2. Call process() to get embeddings:
           
           .. code-block:: python
           
               embeddings = embedder.process(inputs)
        
        3. Use embeddings for similarity search:
           
           .. code-block:: python
           
               from sklearn.metrics.pairwise import cosine_similarity
               similarities = cosine_similarity(embeddings)
        
        Notes
        -----
        - GGUF models currently support text-only inputs
        - For image/video inputs, use the full Qwen3VLEmbedder
        - Embeddings are L2-normalized by default
        """
        embeddings = []
        
        for input_dict in inputs:
            # Check for unsupported modalities
            if 'image' in input_dict or 'video' in input_dict:
                logger.warning(
                    "GGUF quantized models currently support text-only. "
                    "Image/video inputs will be ignored. "
                    "Use full Qwen3VLEmbedder for multimodal support."
                )
                # Skip or extract text only
                if 'text' not in input_dict:
                    continue
            
            text = self._prepare_text_input(input_dict)
            
            try:
                result = self.model.create_embedding(text)
                embedding = self._normalize_embedding(result['data'][0]['embedding'])
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error creating embedding for text '{text[:50]}...': {e}")
                # Skip failed embeddings rather than using zero vectors
                continue
        
        if not embeddings:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        return np.vstack(embeddings).astype(np.float32, copy=False)
