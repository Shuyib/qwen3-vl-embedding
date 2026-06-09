# JEPA + Recursive Self-Improvement Spec
## For Qwen3-VL-Embedding

> **Proposal:** Merge Joint Embedding Predictive Architecture (JEPA) principles into Qwen3-VL-Embedding to create a model that can recursively iterate on its own embeddings — predicting what a better embedding should look like, adjusting, and closing the loop inside latent space.

---

## The Big Idea

Right now Qwen3-VL-Embedding does one pass: input → encode → [EOS] embedding → done.

What if instead it could:
1. Generate an initial embedding
2. *Predict* where that embedding *should* be for ideal cross-modal alignment
3. Move toward that prediction
4. Repeat until stabilized

This is **recursive self-improvement in embedding space** — the model improves its own representations by iterating on them internally, without spawning external agents.

---

## Phase 1: JEPA Predictor Head

### What changes
Add a small predictor network on top of the existing [EOS] embedding. The predictor takes one modality's embedding and tries to predict another modality's embedding of the same concept.

### New file: `src/models/jepa_head.py`

```python
import torch
import torch.nn as nn

class JEPAPredictor(nn.Module):
    """
    Tiny MLP that takes embedding_A and tries to predict embedding_B.
    Learns the *structure* of the shared embedding manifold.
    """
    def __init__(self, embed_dim=2048, hidden_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # helps prevent representation collapse
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)
```

### Loss function: VICReg-style (no negatives needed)

For a batch of paired (image, text) embeddings `z_img`, `z_txt`:

1. **Variance term:** Keep std of each dimension above a threshold `γ` — prevents collapse to all-zeros
2. **Covariance term:** Diagonalize the covariance matrix — decorrelates dimensions, spreads info evenly
3. **Invariance term:** MSE between `predictor(z_img)` and `z_txt` (and vice versa)

```
variance_loss   = max(0, γ - std(z).sqrt())  → per-dimension
covariance_loss = Σᵢ≠ⱼ [Cov(z)]²ᵢⱼ           → off-diagonal
invariance_loss = ‖predictor(z_img) - z_txt‖²  +  ‖predictor(z_txt) - z_img‖²

Total = α * variance + β * covariance + γ * invariance
```

### Modification to `qwen3_vl_embedding.py`

**Add to `Qwen3VLEmbedder.__init__()`:**
```python
self.jepa_predictor = JEPAPredictor(embed_dim=config.hidden_size)
```

**Add to `Qwen3VLEmbedder.process()`:**
```python
embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

# JEPA recursive refinement
if self.jepa_predictor is not None and refine_steps > 0:
    for _ in range(refine_steps):
        predicted = self.jepa_predictor(embeddings)
        embeddings = embeddings + 0.1 * (predicted - embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
```

The predictor acts as a **learned correction operator** — trained to push embeddings toward the "ideal" point in the shared multimodal manifold.

---

## Phase 2: Cross-Modal Training Loop

The key insight: **freeze the base Qwen3-VL encoder** (it already understands images + text). Train only the JEPA predictor head.

### Data format
Your existing data pipeline already works for this:
```python
inputs = [{
    "text": "A woman playing with her dog on a beach at sunset.",
    "image": "https://...demo.jpeg"
}]
```

### New file: `scripts/train_jepa.sh`

```bash
python -m torch.distributed.run --nproc_per_node=4 train_jepa.py \
  --model_name Qwen3-VL-Embedding-2B \
  --data_path ./data/multimodal_pairs/ \
  --freeze_encoder True \
  --jepa_hidden_dim 4096 \
  --lr 2e-5 \
  --num_epochs 3
```

### What happens during training
1. Load paired (image, text) batches
2. Run both through frozen Qwen3-VL → get `z_img`, `z_txt`
3. Run `z_img` through JEPA predictor → get `predicted_txt`
4. Compute VICReg loss between `predicted_txt` and `z_txt`
5. Backprop only through predictor head
6. Repeat

The predictor learns the **geometry of the manifold** — what direction to push an image embedding so it lands near its paired text description (and vice versa).

---

## Phase 3: Recursive Embedding Loop (Inference)

### What changes
`process()` gets a `refine=True` parameter. When on:

```
input → Qwen3VL → embed_0 →
  JEPA predicts better position →
  embed_1 = embed_0 + α·(predicted - embed_0) →
  JEPA predicts again →
  embed_2 → ... → embed_N
```

### Modification to `Qwen3VLEmbedder.process()`

```python
def process(self, inputs, normalize=True, refine=False, refine_steps=5, refine_lr=0.3):
    conversations = [self.format_model_input(...) for ele in inputs]
    processed_inputs = self._preprocess_inputs(conversations)
    processed_inputs = {k: v.to(self.model.device) for k, v in processed_inputs.items()}
    outputs = self.forward(processed_inputs)
    embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

    # Recursive refinement loop
    if refine and self.jepa_predictor is not None:
        for step in range(refine_steps):
            delta = self.jepa_predictor(embeddings) - embeddings
            embeddings = embeddings + refine_lr * delta
            embeddings = F.normalize(embeddings, p=2, dim=-1)

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings
```

### Why this is recursive self-improvement
The model **reads its own embedding → predicts what it should be → moves toward it → repeats**. The improvement comes from internal latent computation. Each loop iteration is a "thought" in embedding space — analogous to chain-of-thought, but happening in the embedding manifold rather than in token space.

---

## Phase 4: Self-Diagnosis (Closing the Loop)

The model identifies regions of embedding space where it's weak, and can flag them for targeted improvement.

### Modification to `Qwen3VLEmbedder`

```python
def diagnose_embedding_health(self, embeddings):
    """Check if embeddings live in a well-structured region of the manifold."""
    with torch.no_grad():
        predicted = self.jepa_predictor(embeddings)
        reconstruction_error = F.mse_loss(predicted, embeddings)
        std_per_dim = embeddings.std(dim=0).mean()
        diversity_score = std_per_dim.item()

        return {
            "reconstruction_error": reconstruction_error.item(),
            "diversity_score": diversity_score,
            "needs_improvement": reconstruction_error > self.health_threshold
        }
```

The trigger: *"This region has high reconstruction error → the manifold isn't well-learned here → flag it for new training data."*

This is where **recursive self-improvement becomes autonomous** — the model tells you what it's bad at, so you can feed it the right examples to get better.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                  Qwen3-VL-Embedding (frozen)                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────────────┐  │
│  │  Image    │  │  Text    │  │  Qwen3VL Backbone            │  │
│  │  Encoder  │  │  Encoder │  │  (visual + language model)   │  │
│  └────┬─────┘  └────┬─────┘  └──────────┬────────────────────┘  │
│       │             │                    │                       │
│       └──────┬──────┘     [EOS] pooling  │                       │
│              ▼                           ▼                       │
│       ┌──────────────┐           ┌──────────────┐               │
│       │  img_embed   │           │  txt_embed   │               │
│       └──────┬───────┘           └──────┬───────┘               │
└──────────────┼──────────────────────────┼───────────────────────┘
               │                          │
               ▼                          ▼
       ┌───────────────────────────────────────────────┐
       │           JEPA Predictor Head                  │
       │  (trained — only module that learns)           │
       │                                                │
       │  img_embed → predict → predicted_txt_embed     │
       │  txt_embed → predict → predicted_img_embed     │
       │                                                │
       │  Loss = α·Var + β·Cov + γ·Invariance           │
       └──────────────────┬────────────────────────────┘
                          │
                          ▼
               ┌──────────────────┐
               │  Recursive Loop  │ ◄──── LOOP BACK (N times)
               │                  │
               │  embed += α ·   │
               │  (predict -     │
               │   embed)        │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Final Embedding │
               │  (stabilized)    │
               └──────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  Self-Diagnosis  │
               │  → health check  │
               │  → flag weak     │
               │    regions       │
               └──────────────────┘
```

---

## Implementation Roadmap

| Phase | What | Effort | Impact |
|-------|------|--------|--------|
| **P1** | JEPA predictor head + VICReg loss module | 1-2 days | Replaces contrastive negative mining; more stable training dynamics |
| **P2** | Cross-modal training loop with frozen backbone | 2-3 days | Learns manifold structure cheaply — no need to retrain Qwen3VL |
| **P3** | Recursive refinement at inference time | 1 day | Embeddings improve by "thinking longer" — more steps = more precision |
| **P4** | Self-diagnosis + data flagging system | 2 days | Closes the loop — model identifies and communicates its own weaknesses |
| **Bonus** | Embedding OS API (manifold walking, interpolation, steering) | varies | Full "embeddings as executable state" vision |

---

## Key File Changes

```
qwen3-vl-embedding/
├── src/
│   └── models/
│       ├── qwen3_vl_embedding.py   ← add refine loop, diagnosis
│       └── jepa_head.py            ← NEW: predictor + VICReg loss
├── scripts/
│   └── train_jepa.sh               ← NEW: training script
├── tests/
│   └── test_jepa.py                ← NEW: unit tests
└── JEPA_RECURSIVE_SPEC.md          ← THIS FILE
```

---

## Related Reading

- **JEPA / I-JEPA:** Yann LeCun's Joint Embedding Predictive Architecture — predicting in latent space instead of pixel space
- **VICReg:** Variance-Invariance-Covariance Regularization — the loss that prevents collapse without negative pairs
- **Ouro / Looped Transformers:** Scaling latent reasoning via looped language models — the "loop in latent space" idea at the architecture level
- **Anthropic Institute Report:** Recursive self-improvement trajectory — AI systems progressively closing the development loop
- **Embedding OS:** The vision of embeddings as first-class OS primitives — compute by navigating meaning-space