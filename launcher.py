#!/usr/bin/env python3
"""
Main entry point for the multimodal file launcher.
"""
import argparse
import logging
import sys
from pathlib import Path

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False

from src.launcher.indexer import FileIndexer
from src.launcher.file_type_detector import make_file_type_detector
from src.launcher.search_engine import SearchEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multimodal File Launcher - Search files using natural language or images"
    )
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-Embedding-2B',
                              help='Model name, path, or GGUF file path')
    model_parser.add_argument('--device', type=str, default='cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu',
                              help='Device to run on (cuda/cpu)')
    model_parser.add_argument('--quantized', action='store_true',
                              help='Use quantized GGUF model (auto-detected for .gguf files)')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', parents=[model_parser], help='Index files in a directory')
    index_parser.add_argument('directory', type=str, help='Directory to index')
    index_parser.add_argument('--no-recursive', action='store_true', help='Do not index subdirectories')
    index_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    index_parser.add_argument(
        '--file-type-detector',
        choices=['auto', 'extension', 'magika'],
        default='auto',
        help='File type detection mode for indexing (default: auto)',
    )
    
    # Launch command
    launch_parser = subparsers.add_parser('launch', parents=[model_parser], help='Launch the search UI')
    launch_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    launch_parser.add_argument('--port', type=int, default=7860, help='Server port')
    launch_parser.add_argument('--share', action='store_true', help='Create a public link')
    launch_parser.add_argument('--keyboard-shortcut', type=str, default=None, 
                              help='Enable keyboard shortcut (e.g., "<ctrl>+<alt>+f")')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the index')
    clear_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show index information')
    info_parser.add_argument('--index-dir', type=str, default='.file_launcher_index', help='Index directory')
    
    return parser.parse_args()


def load_embedder(model_name_or_path: str, device: str, use_quantized: bool = False):
    """
    Load the embedder model (full or quantized).
    
    Parameters
    ----------
    model_name_or_path : str
        Model name, path, or GGUF file path for quantized models.
    device : str
        Device to run on ('cuda' or 'cpu').
    use_quantized : bool, optional
        Whether to use quantized GGUF model (default: False).
    
    Returns
    -------
    embedder : Qwen3VLEmbedder or QuantizedEmbedder
        Loaded embedder instance.
    
    Examples
    --------
    Load full-precision model:
    
    >>> embedder = load_embedder("Qwen/Qwen3-VL-Embedding-2B", "cuda", False)
    
    Load quantized GGUF model:
    
    >>> embedder = load_embedder("./models/model.gguf", "cpu", True)
    
    How to Use
    ----------
    This function automatically detects GGUF files and loads appropriate embedder:
    
    .. code-block:: python
    
        # For GGUF quantized models
        embedder = load_embedder(
            "./models/Qwen3-VL-Embedding-2B-Q4_K_M.gguf",
            device="cpu",
            use_quantized=True
        )
        
        # For full-precision models
        embedder = load_embedder(
            "./models/Qwen3-VL-Embedding-2B",
            device="cuda",
            use_quantized=False
        )
    """
    # Auto-detect GGUF format
    if model_name_or_path.endswith('.gguf'):
        use_quantized = True
    
    if use_quantized:
        logger.info(f"Loading quantized GGUF model: {model_name_or_path}")
        try:
            from src.launcher.quantized_embedder import QuantizedEmbedder
            
            # Determine GPU layers based on device
            n_gpu_layers = 35 if device == 'cuda' else 0
            
            embedder = QuantizedEmbedder(
                model_path=model_name_or_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=8192
            )
            logger.info("Quantized model loaded successfully")
            return embedder
        except Exception as e:
            logger.error(f"Error loading quantized model: {e}")
            logger.error("Make sure llama-cpp-python is installed: pip install llama-cpp-python")
            sys.exit(1)
    else:
        if not EMBEDDER_AVAILABLE or not TORCH_AVAILABLE:
            logger.error("Full-precision embedder requires torch and transformers.")
            logger.error("Install with: pip install torch transformers")
            logger.error("Or use quantized GGUF model with --quantized flag")
            sys.exit(1)
            
        logger.info(f"Loading full-precision embedder model: {model_name_or_path}")
        
        try:
            embedder = Qwen3VLEmbedder(
                model_name_or_path=model_name_or_path,
                torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            )
            logger.info("Model loaded successfully")
            return embedder
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("Please ensure the model is downloaded. See README for instructions.")
            sys.exit(1)


def command_index(args):
    """Handle the index command."""
    embedder = load_embedder(args.model, args.device, args.quantized)
    file_type_detector = make_file_type_detector(args.file_type_detector)
    indexer = FileIndexer(embedder, args.index_dir, file_type_detector=file_type_detector)
    
    logger.info(f"Indexing directory: {args.directory}")
    indexer.index_directory(args.directory, recursive=not args.no_recursive)
    logger.info("Indexing complete!")


def command_launch(args):
    """Handle the launch command."""
    try:
        from src.launcher.ui import LauncherUI
    except ImportError as e:
        logger.error(f"Launcher UI dependencies are not available: {e}")
        logger.error("Install launcher UI dependencies with: uv pip install gradio")
        sys.exit(1)

    embedder = load_embedder(args.model, args.device, args.quantized)
    indexer = FileIndexer(embedder, args.index_dir)
    
    if not indexer.file_metadata:
        logger.error("Index is empty. Please run 'index' command first.")
        sys.exit(1)
    
    search_engine = SearchEngine(embedder, indexer)
    ui = LauncherUI(search_engine)
    
    if args.keyboard_shortcut:
        logger.info(f"Setting up keyboard shortcut: {args.keyboard_shortcut}")
        try:
            from src.launcher.keyboard_handler import UILauncher
            
            launcher = UILauncher(ui)
            handler = launcher.setup_keyboard_shortcut(args.keyboard_shortcut)
            
            # Launch UI immediately
            launcher.launch_ui()
            
            # Start keyboard listener
            handler.start()
            
            logger.info("Press Ctrl+C to exit")
            try:
                handler.run()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                handler.stop()
        except ImportError as e:
            logger.error(f"Keyboard shortcut not available: {e}")
            logger.error("This may require X server or special permissions.")
            logger.info("Launching UI without keyboard shortcut...")
            ui.launch(share=args.share, server_port=args.port)
    else:
        # Just launch the UI
        ui.launch(share=args.share, server_port=args.port)


def command_clear(args):
    """Handle the clear command."""
    # Create a minimal indexer without embedder for clearing
    class NoOpIndexer(FileIndexer):
        def __init__(self, index_dir):
            self.index_dir = Path(index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.index_file = self.index_dir / "file_index.json"
            self.embeddings_file = self.index_dir / "embeddings.npy"
            self.file_metadata = []
            self.embeddings = None
            self.load_index()
    
    indexer = NoOpIndexer(args.index_dir)
    
    confirm = input("Are you sure you want to clear the index? (yes/no): ")
    if confirm.lower() in ['yes', 'y']:
        indexer.clear_index()
        logger.info("Index cleared successfully")
    else:
        logger.info("Clear cancelled")


def command_info(args):
    """Handle the info command."""
    # Create a minimal indexer without embedder for info
    class NoOpIndexer(FileIndexer):
        def __init__(self, index_dir):
            self.index_dir = Path(index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.index_file = self.index_dir / "file_index.json"
            self.embeddings_file = self.index_dir / "embeddings.npy"
            self.file_metadata = []
            self.embeddings = None
            self.load_index()
    
    indexer = NoOpIndexer(args.index_dir)
    
    if not indexer.file_metadata:
        logger.info("Index is empty")
        return
    
    logger.info(f"Index directory: {args.index_dir}")
    logger.info(f"Total files: {len(indexer.file_metadata)}")
    
    # Count by type
    text_count = sum(1 for item in indexer.file_metadata if item['type'] == 'text')
    image_count = sum(1 for item in indexer.file_metadata if item['type'] == 'image')
    
    logger.info(f"Text files: {text_count}")
    logger.info(f"Image files: {image_count}")
    
    # Total size
    total_size = sum(item['size'] for item in indexer.file_metadata)
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Total size: {size_mb:.2f} MB")


def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        logger.error("Please specify a command. Use --help for usage information.")
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == 'index':
        command_index(args)
    elif args.command == 'launch':
        command_launch(args)
    elif args.command == 'clear':
        command_clear(args)
    elif args.command == 'info':
        command_info(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
