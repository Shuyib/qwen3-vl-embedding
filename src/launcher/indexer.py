"""
File indexer module for generating and storing multimodal embeddings.

This module provides functionality to index files in directories using
Qwen3-VL embeddings, supporting both text and image files with efficient
incremental updates and local storage.
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from src.launcher.file_type_detector import (
    AutoFileTypeDetector,
    FileTypeInfo,
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
)

logger = logging.getLogger(__name__)


class FileIndexer:
    """
    Index files in directories using multimodal embeddings.
    
    This class handles scanning directories for supported files, generating
    embeddings using Qwen3-VL models, and maintaining a persistent local index
    with efficient incremental updates.
    
    Parameters
    ----------
    embedder : Qwen3VLEmbedder or QuantizedEmbedder
        Embedder instance for generating embeddings.
    index_dir : str, optional
        Directory to store index files (default: ".file_launcher_index").
    
    Attributes
    ----------
    embedder : object
        The embedder instance.
    index_dir : Path
        Path to the index directory.
    file_metadata : list of dict
        List of file metadata dictionaries.
    embeddings : np.ndarray or None
        Array of embeddings for indexed files.
    
    Examples
    --------
    Basic indexing workflow:
    
    >>> from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    >>> from src.launcher.indexer import FileIndexer
    >>> 
    >>> # Initialize embedder and indexer
    >>> embedder = Qwen3VLEmbedder("./models/Qwen3-VL-Embedding-2B")
    >>> indexer = FileIndexer(embedder, index_dir=".my_index")
    >>> 
    >>> # Index a directory
    >>> indexer.index_directory("/path/to/documents", recursive=True)
    >>> 
    >>> # Check index stats
    >>> print(f"Indexed {len(indexer.file_metadata)} files")
    
    Using quantized embedder:
    
    >>> from src.launcher.quantized_embedder import QuantizedEmbedder
    >>> embedder = QuantizedEmbedder("./models/gguf/model.gguf")
    >>> indexer = FileIndexer(embedder)
    >>> indexer.index_directory("/path/to/code", recursive=True)
    
    How to Use
    ----------
    1. Initialize with an embedder:
       
       .. code-block:: python
       
           from src.launcher.indexer import FileIndexer
           from src.launcher.quantized_embedder import QuantizedEmbedder
           
           embedder = QuantizedEmbedder("model.gguf")
           indexer = FileIndexer(embedder, index_dir=".index")
    
    2. Index files in a directory:
       
       .. code-block:: python
       
           indexer.index_directory(
               "/home/user/documents",
               recursive=True  # Include subdirectories
           )
    
    3. Index is automatically saved and loaded:
       
       .. code-block:: python
       
           # Later, create indexer again - it loads existing index
           indexer2 = FileIndexer(embedder, index_dir=".index")
           print(f"Loaded {len(indexer2.file_metadata)} files")
    
    4. Incremental updates (only new/modified files):
       
       .. code-block:: python
       
           # Add more files - only processes changes
           indexer.index_directory("/home/user/new_docs")
    
    5. Clear the index when needed:
       
       .. code-block:: python
       
           indexer.clear_index()
    
    Notes
    -----
    - Supports text files: .txt, .md, .py, .js, .java, .cpp, .json, etc.
    - Supports image files: .jpg, .png, .gif, .bmp, .webp, .tiff
    - Incremental indexing: Only processes new or modified files
    - Hash-based change detection using file mtime and size
    - Embeddings stored in NumPy format for efficiency
    
    See Also
    --------
    SearchEngine : Search indexed files using queries
    QuantizedEmbedder : Memory-efficient quantized embedder
    """
    
    def __init__(self, embedder, index_dir: str = ".file_launcher_index", file_type_detector=None):
        """
        Initialize the file indexer.
        
        Args:
            embedder: Qwen3VL embedder instance
            index_dir: Directory to store index files
        """
        self.embedder = embedder
        self.file_type_detector = file_type_detector or AutoFileTypeDetector()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / "file_index.json"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        
        self.file_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        self.load_index()
    
    def load_index(self):
        """Load existing index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.file_metadata = json.load(f)
                
                if self.embeddings_file.exists():
                    self.embeddings = np.load(self.embeddings_file)
                    logger.info(f"Loaded index with {len(self.file_metadata)} files")
                else:
                    logger.warning("Index file found but embeddings file missing")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self.file_metadata = []
                self.embeddings = None
    
    def save_index(self):
        """Save index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.file_metadata, f, indent=2)
            
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            logger.info(f"Saved index with {len(self.file_metadata)} files")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def clear_index(self):
        """Clear the entire index."""
        self.file_metadata = []
        self.embeddings = None
        
        if self.index_file.exists():
            self.index_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        
        logger.info("Index cleared")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for change detection."""
        stat = file_path.stat()
        # Use modification time and size as a fast hash
        return hashlib.md5(f"{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
    
    def _detect_file_type(self, file_path: Path) -> FileTypeInfo:
        """Detect file type using the configured detector."""
        return self.file_type_detector.detect(file_path)

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file."""
        return self._detect_file_type(file_path).launcher_type == "text"
    
    def _is_image_file(self, file_path: Path) -> bool:
        """Check if file is an image file."""
        return self._detect_file_type(file_path).launcher_type == "image"
    
    def _read_text_file(self, file_path: Path, max_length: int = 5000) -> str:
        """Read text content from file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_length)
            return content
        except Exception as e:
            logger.warning(f"Error reading text file {file_path}: {e}")
            return ""
    
    def _create_text_preview(self, text: str, max_length: int = 200) -> str:
        """Create a preview snippet of text."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def _create_file_input(self, file_path: Path, file_type: Optional[FileTypeInfo] = None) -> Optional[Dict[str, Any]]:
        """
        Create input dictionary for the embedder based on file type.
        
        Returns:
            Dictionary with appropriate keys for the embedder, or None if unsupported
        """
        file_type = file_type or self._detect_file_type(file_path)

        if file_type.launcher_type == "text":
            content = self._read_text_file(file_path)
            if content:
                context_header = (
                    f"File name: {file_path.name}\n"
                    f"File type: {file_type.description}\n"
                    f"MIME type: {file_type.mime_type}\n\n"
                )
                return {"text": context_header + content}
        elif file_type.launcher_type == "image":
            if getattr(self.embedder, "supports_images", True):
                return {"image": str(file_path.absolute())}
        
        return None
    
    def index_directory(self, directory: str, recursive: bool = True):
        """
        Index all supported files in a directory.
        
        Args:
            directory: Path to directory to index
            recursive: Whether to recursively index subdirectories
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return
        
        # Find all files
        pattern = "**/*" if recursive else "*"
        all_files = list(directory.glob(pattern))
        
        # Filter to supported files
        files_to_index = []
        detected_file_types = {}
        for file_path in all_files:
            if file_path.is_file():
                file_type = self._detect_file_type(file_path)
                if file_type.is_supported:
                    files_to_index.append(file_path)
                    detected_file_types[file_path] = file_type
        
        logger.info(f"Found {len(files_to_index)} supported files to index")
        
        # Check which files need updating
        existing_hashes = {}
        if self.file_metadata:
            existing_hashes = {item['path']: item['hash'] for item in self.file_metadata}
        
        files_needing_update = []
        
        for file_path in files_to_index:
            path_str = str(file_path.absolute())
            current_hash = self._get_file_hash(file_path)
            
            if path_str not in existing_hashes or existing_hashes[path_str] != current_hash:
                files_needing_update.append((file_path, current_hash))
        
        if not files_needing_update:
            logger.info("All files are already indexed and up to date")
            return
        
        logger.info(f"Indexing {len(files_needing_update)} new/modified files")
        
        # Process files in batches
        batch_size = 8
        new_metadata = []
        new_embeddings = []
        
        for i in tqdm(range(0, len(files_needing_update), batch_size), desc="Indexing files"):
            batch = files_needing_update[i:i + batch_size]
            batch_inputs = []
            batch_metadata = []
            
            for file_path, file_hash in batch:
                file_type = detected_file_types[file_path]
                file_input = self._create_file_input(file_path, file_type)
                if file_input:
                    batch_inputs.append(file_input)
                    
                    # Create metadata
                    metadata = {
                        "path": str(file_path.absolute()),
                        "name": file_path.name,
                        "type": file_type.launcher_type,
                        "hash": file_hash,
                        "size": file_path.stat().st_size,
                        "file_type": file_type.to_metadata(),
                    }
                    
                    # Add preview for text files
                    if metadata["type"] == "text":
                        content = self._read_text_file(file_path)
                        metadata["preview"] = self._create_text_preview(content)
                    
                    batch_metadata.append(metadata)
            
            # Generate embeddings for batch
            if batch_inputs:
                try:
                    batch_embeddings = self.embedder.process(batch_inputs)
                    
                    # Convert to numpy if it's a tensor
                    if hasattr(batch_embeddings, 'cpu'):
                        batch_embeddings = batch_embeddings.cpu().numpy()

                    batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
                    if batch_embeddings.ndim == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)

                    if batch_embeddings.shape[0] != len(batch_metadata):
                        logger.error(
                            "Skipping batch: embedder returned %s embeddings for %s files",
                            batch_embeddings.shape[0],
                            len(batch_metadata),
                        )
                        continue

                    if batch_embeddings.shape[0] == 0:
                        continue
                    
                    new_metadata.extend(batch_metadata)
                    new_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        # Update index
        if new_embeddings:
            new_embeddings_array = np.vstack(new_embeddings)
            
            # Remove old entries for updated files
            updated_paths = {item['path'] for item in new_metadata}
            if self.file_metadata:
                kept_indices = [i for i, item in enumerate(self.file_metadata) if item['path'] not in updated_paths]
                self.file_metadata = [self.file_metadata[i] for i in kept_indices]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[kept_indices]
            
            # Add new entries
            self.file_metadata.extend(new_metadata)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
            
            self.save_index()
            logger.info(f"Index updated: {len(self.file_metadata)} total files")
