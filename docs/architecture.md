# Architecture & benchmarks

This page keeps the detailed model reference, architecture notes, input specification, and benchmark tables out of the repository’s quick-start path. Return to the [README](../README.md) to choose an implementation path.

## Model specifications

| Model | Size | Layers | Sequence Length | Embedding Dimension | Quantization Support | Matryoshka Representation Learning | Instruction Aware |
|---|---:|---:|---:|---:|---|---|---|
| Qwen3-VL-Embedding-2B | 2B | 28 | 32K | 2048 | Yes | Yes | Yes |
| Qwen3-VL-Embedding-8B | 8B | 36 | 32K | 4096 | Yes | Yes | Yes |
| Qwen3-VL-Reranker-2B | 2B | 28 | 32K | — | — | — | Yes |
| Qwen3-VL-Reranker-8B | 8B | 36 | 32K | — | — | — | Yes |

### Model sources

| Model | Hugging Face | ModelScope |
|---|---|---|
| Qwen3-VL-Embedding-2B | [Download](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | [Download](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-2B) |
| Qwen3-VL-Embedding-8B | [Download](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | [Download](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-8B) |
| Qwen3-VL-Reranker-2B | [Download](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | [Download](https://modelscope.cn/models/qwen/Qwen3-VL-Reranker-2B) |
| Qwen3-VL-Reranker-8B | [Download](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) | [Download](https://modelscope.cn/models/qwen/Qwen3-VL-Reranker-8B) |

### LoRA configuration

| Model | Rank | Alpha | Target modules |
|---|---:|---:|---|
| Qwen3-VL-Embedding | 32 | 32 | `q_proj`, `v_proj`, `k_proj`, `up_proj`, `down_proj`, `gate_proj` |
| Qwen3-VL-Reranker | 32 | 32 | `q_proj`, `v_proj`, `k_proj`, `up_proj`, `down_proj`, `gate_proj` |

## Retrieval architecture

### Embedding model: dual tower

The embedding model accepts a single-modal or mixed-modal item and maps it to a high-dimensional semantic vector. It uses the hidden state for the `[EOS]` token from the final base-model layer as the semantic representation. Because queries and documents are encoded independently, it is suitable for large-scale vector retrieval.

### Reranker model: single tower

The reranker accepts a `(query, document)` pair and computes a pointwise relevance score. Cross-attention allows finer-grained interaction and fusion across text, images, and video. Relevance is expressed through the generation probability of special `yes` and `no` tokens.

| | Embedding | Reranker |
|---|---|---|
| Core function | Semantic representation and vector generation | Relevance scoring and pointwise reranking |
| Input | One single- or mixed-modal item | One query/document pair |
| Architecture | Dual tower | Single tower |
| Mechanism | Efficient similarity retrieval | Deep inter-modal interaction and precise alignment |
| Output | Semantic vector | Relevance score |

## Input specification

A multimodal object may include any combination of:

- `text`: a string or list of strings;
- `image`: local paths, URLs, `PIL.Image.Image` instances, or a list of these;
- `video`: local paths, URLs, sequences of image frames, or a list of these.

All modalities support either one value or a list of values. The optional `instruction` describes the relevance task and defaults to `Represent the user's input`. For video files, `fps` controls sampling rate and `max_frames` caps sampled frames.

Embedding input is a list of dictionaries. Reranking input is one dictionary containing `query`, `documents`, optional `instruction`, `fps`, and `max_frames`.

### Embedding initialization parameters

```python
Qwen3VLEmbedder(
    model_name_or_path="./models/Qwen3-VL-Embedding-2B",
    max_length=8192,
    min_pixels=4096,
    max_pixels=1843200,
    total_pixels=7864320,
    fps=1.0,
    max_frames=64,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

## Benchmark results

### Embedding: MMEB-V2

Results on [MMEB-V2](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard). CLS: classification; QA: question answering; RET: retrieval; GD: grounding; MRET: moment retrieval; VDR: ViDoRe; VR: VisRAG; OOD: out-of-distribution.

| Model | Model Size | Image CLS | Image QA | Image RET | Image GD | Image Overall | Video CLS | Video QA | Video RET | Video MRET | Video Overall | VisDoc VDRv1 | VisDoc VDRv2 | VisDoc VR | VisDoc OOD | VisDoc Overall | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-Embedding-2B | 2B | 70.3 | 74.3 | 74.8 | 88.5 | 75.0 | 71.9 | 64.9 | 53.9 | 53.3 | 61.9 | 84.4 | 65.3 | 86.4 | 69.4 | 79.2 | 73.2 |
| Qwen3-VL-Embedding-8B | 8B | 74.2 | 81.1 | 80.0 | 92.2 | 80.1 | 78.4 | 71.0 | 58.7 | 67.1 | 67.1 | 87.2 | 69.9 | 88.7 | 73.3 | 82.4 | 77.8 |

For the complete upstream comparison tables, see the [technical report](../assets/qwen3vlembedding_technical_report.pdf). Benchmark numbers should be interpreted with the corresponding evaluation configuration.

### Reranking

The reranker is evaluated on MMEB-v2 retrieval, MMTEB retrieval, JinaVDR, and ViDoRe v3. It improves candidate ordering compared with embedding-only retrieval, especially when used as a second stage.

| Model | Size | MMEB-v2 Avg | Image | Video | VisDoc | MMTEB | JinaVDR | ViDoRe v3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-Embedding-2B | 2B | 73.4 | 74.8 | 53.6 | 79.2 | 68.1 | 71.0 | 52.9 |
| jina-reranker-m0 | 2B | — | 68.2 | — | 85.2 | — | 82.2 | 57.8 |
| Qwen3-VL-Reranker-2B | 2B | 75.2 | 74.0 | 53.2 | 83.2 | 70.0 | 80.9 | 60.8 |
| Qwen3-VL-Reranker-8B | 8B | 79.2 | 78.2 | 61.0 | 85.8 | 74.9 | 83.6 | 66.7 |

## Evaluation

```bash
bash data/evaluation/mmeb_v2/download_data.sh
bash scripts/evaluation/mmeb_v2/eval_embedding.sh
bash scripts/evaluation/mmeb_v2/eval_reranker.sh
```
