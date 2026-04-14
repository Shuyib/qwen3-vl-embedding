# Multimodal File Launcher

A keyboard-launchable multimodal file search tool using Qwen3-VL embeddings. Search for files by describing their content in natural language or by providing a reference image, rather than relying on filenames or paths.

## Features

- **🎨 Multimodal Search**: Search using natural language descriptions or reference images
- **⌨️ Keyboard Shortcuts**: Global keyboard shortcut for quick access (System76-style)
- **🔍 Semantic Understanding**: Powered by Qwen3-VL embeddings for deep content understanding
- **📁 Offline Indexing**: Build local indexes of your files with multimodal embeddings
- **🖼️ Visual Previews**: Lightweight previews with text snippets or image thumbnails
- **🎯 High Accuracy**: Semantic similarity search returning the most relevant files

## Installation

The multimodal file launcher is included in the main package. Install dependencies:

```bash
# Install dependencies
uv pip install gradio pynput pillow faiss-cpu tqdm llama-cpp-python
```

Or update all dependencies:
```bash
bash scripts/setup_environment.sh
```

### Optional: Magika File Type Detection

Install Magika if you want the launcher to detect file types from file content instead of relying only on file extensions:

```bash
uv pip install magika
```

With Magika available, `--file-type-detector auto` can index supported extensionless or misnamed files such as `Dockerfile`, shell scripts, source files, and configuration files. Magika is used for detection and metadata enrichment; rich document text extraction for formats like PDF or DOCX still requires a separate extractor.

### Download Models

You can use either full-precision or quantized models:

**Full-Precision Models (Multimodal Support):**
```bash
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./models/Qwen3-VL-Embedding-2B
```

**Quantized GGUF Models (Text-Only, Memory Efficient):**
```bash
# Download quantized model (Q4_K_M is recommended for balance of size and quality)
huggingface-cli download DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF \
    Qwen3-VL-Embedding-2B-Q4_K_M.gguf \
    --local-dir ./models/gguf/
```

**Quantization Options:**
- `Q4_K_M`: Recommended - good balance (4-bit, ~1.5GB)
- `Q5_K_M`: Higher quality (5-bit, ~1.8GB)
- `Q8_0`: Best quality (8-bit, ~2.5GB)
- See [DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF) for all options

## Quick Start

### 1. Index Your Files

First, index a directory containing files you want to search:

**Using Full-Precision Model (supports images):**
```bash
python launcher.py index /path/to/your/documents --model ./models/Qwen3-VL-Embedding-2B
```

**Using Magika for content-based file type detection:**
```bash
python launcher.py index /path/to/your/documents \
    --model ./models/Qwen3-VL-Embedding-2B \
    --file-type-detector magika
```

**Using Quantized GGUF Model (text-only, memory efficient):**
```bash
python launcher.py index /path/to/your/documents --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf --quantized
```

This will:
- Scan the directory for supported files (text and images)
- Generate multimodal embeddings for each file
- Store the index locally in `.file_launcher_index/`

Supported file types:
- **Text files**: `.txt`, `.md`, `.py`, `.js`, `.java`, `.cpp`, `.json`, `.yaml`, `.html`, `.css`, etc.
- **Image files**: `.jpg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff` (full-precision model only)

### 2. Launch the Search UI

Launch the Gradio interface:

**Using Full-Precision Model:**
```bash
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B
```

**Using Quantized GGUF Model (recommended for CPU):**
```bash
python launcher.py launch --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf --quantized --device cpu
```

The UI will open in your browser at `http://localhost:7860`

### 3. Search Your Files

**Text Search Tab:**
- Enter a natural language description of what you're looking for
- Example: "Python code for machine learning"
- Example: "Configuration file with database settings"

**Image Search Tab:**
- Upload a reference image
- Find similar images or related content

## Keyboard Shortcut Mode

Launch with a global keyboard shortcut (Ctrl+Alt+F by default):

```bash
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B --keyboard-shortcut "<ctrl>+<alt>+f"
```

Now press `Ctrl+Alt+F` from anywhere on your system to open the search interface!

**Note**: Global keyboard shortcuts may require special permissions on some systems (e.g., accessibility permissions on macOS).

## Command Reference

### Index Command

Index files in a directory:

```bash
python launcher.py index <directory> [options]

Options:
  --no-recursive          Do not index subdirectories
  --index-dir DIR        Custom index directory (default: .file_launcher_index)
  --file-type-detector   File type detection mode: auto, extension, or magika
  --model MODEL          Model name, path, or GGUF file (default: Qwen/Qwen3-VL-Embedding-2B)
  --device DEVICE        Device to use: cuda or cpu (default: auto-detect)
  --quantized            Use quantized GGUF model (auto-detected for .gguf files)
```

File type detection modes:
- `auto`: use Magika when installed; otherwise fall back to extension-based detection.
- `extension`: use the launcher's built-in extension allow-list only.
- `magika`: require Magika and fail fast if it is not installed.

### Launch Command

Launch the search UI:

```bash
python launcher.py launch [options]

Options:
  --index-dir DIR               Index directory (default: .file_launcher_index)
  --port PORT                   Server port (default: 7860)
  --share                       Create a public link
  --keyboard-shortcut SHORTCUT  Enable keyboard shortcut (e.g., "<ctrl>+<alt>+f")
  --model MODEL                 Model name, path, or GGUF file
  --device DEVICE               Device to use: cuda or cpu
  --quantized                   Use quantized GGUF model
```

### Info Command

Show index information:

```bash
python launcher.py info [options]

Options:
  --index-dir DIR        Index directory (default: .file_launcher_index)
```

### Clear Command

Clear the index:

```bash
python launcher.py clear [options]

Options:
  --index-dir DIR        Index directory (default: .file_launcher_index)
```

## Usage Examples

### Example 1: Index and Search Code Repository

```bash
# Index your code repository
python launcher.py index ~/projects/my-app --model ./models/Qwen3-VL-Embedding-2B

# Launch with keyboard shortcut
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B --keyboard-shortcut "<ctrl>+<alt>+f"

# Search examples:
# - "authentication logic"
# - "database connection code"
# - "configuration files"
```

### Example 2: Search Document Collection

```bash
# Index documents
python launcher.py index ~/Documents --model ./models/Qwen3-VL-Embedding-2B

# Launch UI
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B

# Search examples:
# - "budget report for Q4"
# - "meeting notes about the project"
# - "invoice from last month"
```

### Example 3: Image Collection Search

```bash
# Index photo library
python launcher.py index ~/Pictures --model ./models/Qwen3-VL-Embedding-2B

# Launch UI
python launcher.py launch --model ./models/Qwen3-VL-Embedding-2B

# Search with:
# - Text: "sunset on the beach"
# - Image: Upload a reference photo to find similar images
```

## Architecture

The multimodal file launcher consists of several integrated components working together to provide semantic file search:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐        ┌──────────────────────────────────┐  │
│  │  CLI (launcher.py)│        │   Gradio UI (ui.py)              │  │
│  │  • index          │───────▶│   • Text Search Tab              │  │
│  │  • launch         │        │   • Image Search Tab             │  │
│  │  • info/clear     │        │   • Results Display              │  │
│  └──────────────────┘        └──────────────────────────────────┘  │
│           │                             │                            │
│           │                             │                            │
│           │    ┌────────────────────────┘                            │
│           │    │                                                     │
└───────────┼────┼─────────────────────────────────────────────────────┘
            │    │
            ▼    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Core Application Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────┐    ┌───────────────────────────┐   │
│  │  File Indexer (indexer.py) │    │ Search Engine             │   │
│  │  • Directory Scanning      │    │ (search_engine.py)        │   │
│  │  • File Type Detection     │───▶│ • Query Processing        │   │
│  │  • Content Extraction      │    │ • Similarity Calculation  │   │
│  │  • Batch Processing        │    │ • Result Ranking          │   │
│  │  • Incremental Updates     │    │ • Top-K Selection         │   │
│  └────────────────────────────┘    └───────────────────────────┘   │
│            │                                  ▲                      │
│            │                                  │                      │
│            ▼                                  │                      │
│  ┌─────────────────────────────────────────┐ │                      │
│  │     Local Index Storage                 │ │                      │
│  │  • file_index.json (metadata)           │─┘                      │
│  │  • embeddings.npy (vectors)             │                        │
│  │  • Hash-based change detection          │                        │
│  └─────────────────────────────────────────┘                        │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Embedding Layer                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐  │
│  │  Qwen3VLEmbedder         │    │  QuantizedEmbedder           │  │
│  │  (Full Model)            │    │  (GGUF/llama-cpp)            │  │
│  │  • Text Embeddings       │    │  • Text-Only Embeddings      │  │
│  │  • Image Embeddings      │    │  • 4-8x Memory Reduction     │  │
│  │  • Video Embeddings      │    │  • CPU Optimized             │  │
│  │  • GPU Acceleration      │    │  • Q4_K_M, Q5_K_M, Q8_0      │  │
│  └──────────────────────────┘    └──────────────────────────────┘  │
│            │                                  │                      │
│            └──────────────┬───────────────────┘                      │
│                           │                                          │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Qwen3-VL Base  │
                   │     Models      │
                   │  (HuggingFace)  │
                   └─────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Optional: Keyboard Shortcuts                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  KeyboardShortcutHandler (keyboard_handler.py)             │    │
│  │  • Global Hotkey Listener (Ctrl+Alt+F)                     │    │
│  │  • UI Launcher with Thread Safety                          │    │
│  │  • Graceful Fallback (no X server)                         │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

**1. File Indexer** (`src/launcher/indexer.py`)
   - Scans directories recursively for supported file types
   - Extracts content from text files and metadata from images
   - Generates embeddings via the embedding layer
   - Stores results in local JSON + NumPy format
   - Uses hash-based change detection for incremental updates

**2. Search Engine** (`src/launcher/search_engine.py`)
   - Processes text or image queries through embedder
   - Computes cosine similarity between query and indexed embeddings
   - Ranks results by similarity score
   - Returns top-K most relevant files with metadata

**3. UI Layer** (`src/launcher/ui.py`)
   - Gradio-based web interface with dual tabs
   - Text search: natural language query input
   - Image search: reference image upload
   - Results display with previews and similarity scores

**4. Keyboard Handler** (`src/launcher/keyboard_handler.py`)
   - Optional global keyboard shortcut support
   - Thread-safe UI launching with mutex protection
   - Gracefully handles environments without X server

**5. Embedding Layer**
   - **Qwen3VLEmbedder**: Full-precision multimodal model
   - **QuantizedEmbedder**: Memory-efficient GGUF wrapper
   - Auto-selection based on model file format

### Data Flow

1. **Indexing Flow**:
   ```
   User Files → File Indexer → Embedder → Local Index
   ```

2. **Search Flow**:
   ```
   User Query → Search Engine → Embedder → Similarity Calc → Ranked Results
                                             ↓
                                       Local Index
   ```

3. **UI Flow**:
   ```
   Browser ↔ Gradio UI ↔ Search Engine ↔ Embedder
                              ↓
                         Local Index
   ```

## Performance Tips

- **GPU Acceleration**: Use CUDA-enabled GPU for faster indexing and search with full models
- **Quantized Models**: Use GGUF quantized models for 4-8x memory reduction with minimal accuracy loss
  - Best for CPU-only systems or limited GPU memory
  - Q4_K_M recommended for general use
  - Text-only support (no images/videos)
- **Batch Size**: Large directories are processed in batches automatically
- **Incremental Updates**: Re-indexing only processes new or modified files
- **File Type Detection**: Install Magika and use `--file-type-detector auto` to improve routing for extensionless or mislabeled files
- **Model Selection**: 
  - Use quantized 2B model for fastest CPU inference
  - Use full 2B model for balanced performance with multimodal support
  - Use 8B model for best accuracy (requires more GPU memory)

## Troubleshooting

### Issue: Global keyboard shortcut not working

- On Linux: May require X11 or appropriate Wayland permissions
- On macOS: Grant accessibility permissions in System Preferences
- On Windows: May require running as administrator
- Fallback: Use the launch command without keyboard shortcut

### Issue: Out of memory during indexing

- Use CPU instead of GPU: `--device cpu`
- Index smaller directories separately
- Use the 2B model instead of 8B

### Issue: Slow search performance

- Ensure GPU is being used if available
- Consider reducing the number of indexed files
- Use smaller top_k values (fewer results)

## Quantized Models (GGUF)

The launcher supports quantized GGUF models for memory-efficient CPU inference.

### Benefits of Quantized Models

- **4-8x smaller memory footprint**: ~1.5GB vs ~6GB for 2B model
- **Faster CPU inference**: Optimized for CPU with no GPU required
- **Easy deployment**: Single file, no complex dependencies
- **Minimal accuracy loss**: Q4_K_M maintains >95% quality

### Limitations

- **Text-only**: GGUF models currently support text inputs only
- **No multimodal**: Cannot process images or videos
- For image search, use full-precision Qwen3VLEmbedder

### Download and Usage

Download quantized model:
```bash
huggingface-cli download DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF \
    Qwen3-VL-Embedding-2B-Q4_K_M.gguf \
    --local-dir ./models/gguf/
```

Use in launcher (automatically detected):
```bash
# Index with quantized model
python launcher.py index /path/to/docs \
    --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf

# Launch with quantized model  
python launcher.py launch \
    --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf \
    --device cpu
```

### Quantization Levels

| Quantization | Size | Quality | Use Case |
|-------------|------|---------|----------|
| Q4_K_M | ~1.5GB | 95% | **Recommended** - Best balance |
| Q5_K_M | ~1.8GB | 97% | Higher quality, slightly larger |
| Q8_0 | ~2.5GB | 99% | Near full quality |

## Limitations (Current MVP)

- No cloud sync or multi-device support
- No file permissions or sandboxing
- No full OS integration (e.g., system search integration)
- Limited to local file access

## Future Enhancements

Planned features for future releases:
- System-wide integration with file managers
- Cloud sync support
- Real-time file watching and auto-indexing
- Advanced filtering options
- Custom file type handlers
- Multi-index support

## License

This component follows the same Apache 2.0 license as the main Qwen3-VL-Embedding project.
