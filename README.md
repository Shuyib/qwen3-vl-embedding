<p align="center">
    <img src="https://model-demo.oss-cn-hangzhou.aliyuncs.com/Qwen3-VL-Embedding.png" width="400"/>
    <img src="https://model-demo.oss-cn-hangzhou.aliyuncs.com/Qwen3-VL-Reranker.png" width="400"/>
</p>

# Qwen3-VL-Embedding & Qwen3-VL-Reranker

[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/QwenLM/Qwen3-VL-Embedding)
[![Hugging Face - Embedding](https://img.shields.io/badge/🤗-Embedding-yellow)](https://huggingface.co/collections/Qwen/qwen3-vl-embedding)
[![Hugging Face - Reranker](https://img.shields.io/badge/🤗-Reranker-yellow)](https://huggingface.co/collections/Qwen/qwen3-vl-reranker)
[![Technical Report](https://img.shields.io/badge/📄-Technical%20Report-red)](assets/qwen3vlembedding_technical_report.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Community-enhanced fork for production-minded multimodal retrieval: local semantic file search, Magika content-based file detection, a Gradio launcher, and a stabilized Gradio import path for environments with problematic optional PyArrow builds.**

Qwen3-VL-Embedding turns text, images, screenshots, and video into a shared vector space. Qwen3-VL-Reranker then scores the most promising query/document pairs more precisely. This fork makes that capability easier to try locally, without hiding the original model implementation and evaluation tooling.

## Choose your path

| I want to… | Start here |
|---|---|
| Search my files locally | [Multimodal File Launcher](#multimodal-file-launcher) |
| Build embedding or reranking into an application | [Python model usage](#use-the-models-in-python) |
| Run end-to-end retrieval + reranking examples | [Interactive examples](#interactive-examples) |
| Understand model design, input formats, or benchmark results | [Architecture & benchmarks](docs/architecture.md) |
| Reproduce model evaluations | [Evaluation](#evaluation) |

## Why this fork

- **Local multimodal file search:** index files once, then search by meaning from a Gradio UI or keyboard shortcut.
- **Content-based detection:** optional [Magika](https://github.com/google/magika) detection recognizes supported extensionless or misnamed text files, including Dockerfiles and scripts.
- **Practical model choices:** use a small GGUF model for CPU text search, or the full-precision embedding model for images, video, and mixed-modal inputs.
- **A complete retrieval funnel:** the repository includes both the embedding and reranker implementations, plus notebooks for each. The launcher currently performs vector retrieval; reranking is available to application code and notebooks, not yet wired into the launcher UI.

## The retrieval funnel

```text
Files / query
    │
    ├─► File-type detection (extension rules or Magika)
    ├─► Qwen3-VL Embedding ─► local vectors ─► top-K similarity retrieval
    │                                                   │
    │                                                   └─► Launcher results
    │
    └─► Optional Qwen3-VL Reranker ─► reordered top-K results
```

Use embeddings for fast recall across a large collection. Use the reranker only on the small candidate set when relevance quality matters most.

## Installation

```bash
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
cd Qwen3-VL-Embedding
bash scripts/setup_environment.sh
source .venv/bin/activate
```

Download the model appropriate to your path:

| Path | Model | Best for |
|---|---|---|
| CPU text search | [GGUF embedding model](https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF) | low-memory local search |
| Full multimodal embedding | [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | images, video, and mixed-modal retrieval |
| Reranking candidates | [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | final relevance scoring |

Full model download example:

```bash
uv pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B \
    --local-dir ./models/Qwen3-VL-Embedding-2B
```

See [Architecture & benchmarks](docs/architecture.md) for all supported model sizes, inputs, and providers.

## Multimodal File Launcher

The launcher is the quickest way to experience the repository: describe a file’s content in natural language, or use a reference image, instead of relying on filenames.

### Fast local text search (CPU)

```bash
hf download DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF \
    Qwen3-VL-Embedding-2B-Q4_K_M.gguf \
    --local-dir ./models/gguf/

python launcher.py index ~/Documents \
    --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf

python launcher.py launch \
    --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf
```

The GGUF route is text-only. Use the full-precision model when your index or query includes images or video:

```bash
python launcher.py index ~/Pictures \
    --model ./models/Qwen3-VL-Embedding-2B

python launcher.py launch \
    --model ./models/Qwen3-VL-Embedding-2B
```

### Make file detection more robust

```bash
uv pip install magika
python launcher.py index ~/Documents \
    --model ./models/gguf/Qwen3-VL-Embedding-2B-Q4_K_M.gguf \
    --file-type-detector auto
```

`auto` uses Magika when available and otherwise falls back to extension detection. It improves coverage for supported extensionless or misnamed files; it does not add rich text extraction for PDF or DOCX files.

Run `python demo_launcher.py` for a small local demonstration.

### Choose your launcher guide

| Need | Guide |
|---|---|
| Complete setup, CLI options, keyboard shortcuts, supported types, and troubleshooting | [Launcher reference](LAUNCHER_README.md) |
| Tested commands, real-model results, detection experiments, and practical retrieval caveats | [Launcher cheatsheet](MULTIMODAL_FILE_LAUNCHER_CHEATSHEET.md) |

## Use the models in Python

### Embed a multimodal collection

```python
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

embedder = Qwen3VLEmbedder("./models/Qwen3-VL-Embedding-2B")
vectors = embedder.process([
    {"text": "A woman playing with her dog on a beach at sunset.",
     "instruction": "Retrieve images or text relevant to the user's query."},
    {"image": "./data/examples/0.jpeg"},
])
```

### Rerank retrieved candidates

```python
from src.models.qwen3_vl_reranker import Qwen3VLReranker

reranker = Qwen3VLReranker("./models/Qwen3-VL-Reranker-2B")
scores = reranker.process({
    "instruction": "Retrieve images or text relevant to the user's query.",
    "query": {"text": "A woman playing with her dog on a beach at sunset."},
    "documents": [
        {"text": "A woman and her golden retriever play on a beach at sunset."},
        {"image": "./data/examples/0.jpeg"},
    ],
})
```

For a production flow, retrieve a generous top-K with the full-precision embedding model, then pass those candidates to the reranker. The [multimodal RAG notebook](examples/Qwen3VL_Multimodal_RAG.ipynb) shows the two models in the same workflow.

## Interactive examples

- [Embedding notebook](examples/embedding.ipynb)
- [Embedding with vLLM](examples/embedding_vllm.ipynb)
- [Reranker notebook](examples/reranker.ipynb)
- [Reranker with vLLM](examples/reranker_vllm.ipynb)
- [End-to-end multimodal RAG](examples/Qwen3VL_Multimodal_RAG.ipynb)

## Evaluation

The included scripts reproduce MMEB v2 evaluation for both stages:

```bash
bash data/evaluation/mmeb_v2/download_data.sh
bash scripts/evaluation/mmeb_v2/eval_embedding.sh
bash scripts/evaluation/mmeb_v2/eval_reranker.sh
```

Benchmark tables, model specifications, input-format details, and architectural notes live in [docs/architecture.md](docs/architecture.md).

## Citation

```bibtex
@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen, Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv},
  year={2026}
}
```
