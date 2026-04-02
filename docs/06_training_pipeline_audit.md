# Training Pipeline Audit — What We Have, What's Wrong, What's Needed
> Date: 2026-03-20 | Auditor: Antigravity | Scope: Training only (not comparison/inference)

---

## Current State Summary

### What exists:
- `src/training/muon.py` — Muon optimizer implementation
- `scripts/train_sft.py` — QLoRA SFT training with AdamW vs Muon ablation  
- `scripts/build_sft_dataset.py` — Filters correct RLM trajectories → SFT dataset
- `scripts/eval_sft.py` — Evaluates trained LoRA adapters on NIAH
- `experiments/sft_dataset.jsonl` — 16 examples from 52 trajectories (very small)
- All training dependencies installed (torch, transformers, peft, trl, bitsandbytes, accelerate)

### What has NOT been run:
- Actual training has never happened — scripts exist but `train_sft.py` was never executed
- No checkpoints exist
- No training metrics exist
- The Muon vs AdamW comparison has never been tested
- The `_CombinedOptimizer` wrapper has never been tested with HF Trainer

---

## Issue 1: Muon Implementation — Needs Fixes

### Current code review (src/training/muon.py):

**What's correct ✓:**
- Newton-Schulz iteration coefficients `(3.4445, -4.7750, 2.0315)` — match official Muon
- 5-step default — matches best practice
- Nesterov momentum — correct, matches official implementation
- Spectral norm normalization before NS iterations — correct
- Transpose-if-tall logic — correct

**What needs fixing ✗:**

#### Fix 1: Missing momentum warmup
The official Muon repo recommends **momentum warmup** (0.85 → 0.95 over ~300 steps) to prevent instability in early training. Our implementation uses a fixed 0.95 from step 0.

```python
# Current (wrong):
buf.mul_(momentum).add_(g)

# Should add: momentum schedule that ramps 0.85 → 0.95 over first 300 steps
```

#### Fix 2: Missing scale adjustment for Muon LR
With QLoRA, the trainable parameter count is tiny (0.5-1% of model). Muon's default LR of 0.02 was designed for full-parameter training of NanoGPT/small transformers. For QLoRA LoRA adapters, this may need scaling.

**Current**: `muon_lr = 0.02` (could be too aggressive for LoRA)  
**Recommended**: Start with `muon_lr = 0.005` for LoRA, sweep `[0.002, 0.005, 0.01, 0.02]`

#### Fix 3: Embedding and LM head exclusion  
`split_params_for_muon()` currently splits purely on `ndim == 2`. This incorrectly assigns the embedding matrix and lm_head (which ARE 2D) to Muon. Official guidance says embeddings + lm_head should go to AdamW.

```python
# Current (wrong) — puts all 2D params in Muon:
if param.ndim == 2:
    muon_params.append(param)

# Should exclude embed_tokens and lm_head:
if param.ndim == 2 and not any(x in name for x in ("embed_tokens", "lm_head")):
    muon_params.append(param)
```

With QLoRA though, embed_tokens and lm_head are frozen (not trainable). So this may not matter in practice — BUT we should verify and add the guard anyway for correctness.

---

## Issue 2: _CombinedOptimizer — Compatibility Risk

```python
class _CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, opt1, opt2):
        # We don't call super().__init__() — this is a shim
```

**Problem**: HF Trainer expects certain Optimizer interface properties that this shim may not expose:
- `state` dict aggregation (for checkpoint saving/loading)
- `param_groups` sync with LR scheduler
- `add_param_group()` method

**Risk**: Training may crash on checkpoint save, or LR scheduler may fail silently.

**Fix**: Either:
1. Test the shim carefully with a dry-run (1 step, save checkpoint, resume)
2. OR use TRL's built-in `create_optimizer()` hook override instead of monkey-patching `trainer.optimizer`

---

## Issue 3: Loss Function — Currently Default (Correct, but worth confirming)

**Current**: SFTTrainer uses default **cross-entropy loss** (next-token prediction). This is correct for SFT.

**Loss function hierarchy for our project:**

| Phase | Loss | Notes |
|---|---|---|
| SFT (current) | Cross-entropy (default) | ✓ Correct — learning to copy good trajectories |
| GRPO/RL (future Phase 4c) | Reward-weighted policy gradient | Needs reward function |
| DPO (alternative to GRPO) | DPO loss (preference pairs) | Needs paired data (good vs bad trajectories) |

**No change needed for SFT.** For RL phase, we'll need to define a reward function:
- `+1.0` if final answer is correct
- `-0.5` if REPL had errors
- `+0.2` per valid sub-call (encourages recursion)

---

## Issue 4: SFT Dataset Too Small (16 examples)

**Current dataset**: 16 clean examples from 52 total trajectories.  
**Problem**: This is **way too few** for meaningful SFT. QLoRA best practice says 500–1000 minimum.

**Root cause**: Previous RLM runs had calling issues (premature FINAL, no real sub-calls). The fixes in Session 10 (recursive-first policy) were never verified with live Ollama.

**Action plan:**
1. First: Re-run ALL benchmarks with the fixed RLM runtime (recursive-first + helpers)
2. Use `--save-trajectories` on all runs
3. Target: 200+ correct trajectories before training
4. We need: diverse context lengths (4K–128K), diverse tasks (NIAH + LongDocQA)

---

## Issue 5: Hyperparameter Recommendations (Based on Research)

### For AdamW baseline:
| Param | Current | Recommended | Reason |
|---|---|---|---|
| lr | 2e-4 | 2e-4 | ✓ Correct for QLoRA |
| weight_decay | 0.01 | 0.01 | ✓ Standard |
| warmup_ratio | 0.1 | 0.1 | ✓ Good |
| scheduler | cosine | cosine | ✓ Best for LLMs |
| batch_size | 1 | 1 | ✓ VRAM constraint |
| grad_accum | 4 | 4–8 | Consider 8 for more stable gradients |
| max_seq_length | 2048 | 2048 | ✓ Good for REPL code sequences |
| lora_rank | 8 | 16 | 16 is better for complex tasks (code gen). r=8 may underfit |
| lora_alpha | 16 | 32 | Standard practice: alpha = 2*rank |
| lora_dropout | 0.05 | 0.05 | ✓ Correct |

### For Muon:
| Param | Current | Recommended | Reason |
|---|---|---|---|
| muon_lr | 0.02 | 0.005 | Start lower for LoRA; 0.02 was designed for full params |
| momentum | 0.95 | 0.85→0.95 warmup | Prevents early instability |
| ns_steps | 5 | 5 | ✓ Correct per official repo |
| nesterov | True | True | ✓ Correct |
| weight_decay | 0.0 | 0.01 | Should match AdamW for fair comparison |

### Missing features we should add:
1. **Training loss logging to JSONL** — ✓ Already there (`training_metrics.jsonl`)
2. **Validation split** — ✗ Missing. Should hold out 10-20% for val loss monitoring
3. **TensorBoard / wandb** — ✗ Currently `report_to="none"`. Should add at least TensorBoard
4. **Checkpoint resume** — ✓ Already there (`--resume` flag)
5. **Model merging** — ✗ Missing post-training merge script (`peft.merge_and_unload()`)
6. **Training curves plot** — ✗ Missing. Need a script to plot loss curves from metrics JSONL
7. **Early stopping** — ✗ Missing. Should stop if val loss plateaus
8. **Gradient clipping** — ✗ Missing. Important for Muon stability. Add `max_grad_norm=1.0`

---

## Issue 6: Model ID Mismatch

```python
MODEL_ID = "Qwen/Qwen2.5-1.5B"  # HF model matching qwen3.5:2b capabilities
# Note: Qwen3.5-2B may not be on HF yet. Using Qwen2.5-1.5B as fallback.
```

**Problem**: We're training on Qwen2.5-1.5B but running inference on qwen3.5:2b via Ollama. These are **different models** with different tokenizers and architectures. The LoRA adapter won't be compatible.

**Fix**: Either:
- Train on the actual Qwen3.5 model when available on HuggingFace
- OR run inference with the same Qwen2.5-1.5B model via HuggingFace transformers (not Ollama)
- OR check if `Qwen/Qwen3-2B` exists on HuggingFace now and switch to that

---

## What the Training Pipeline Should Look Like (Complete)

```
Phase A: Generate Training Data
  ├── Fix RLM calling issues (re-run with Session 10 fixes)
  ├── Run diverse benchmarks with --save-trajectories
  ├── Target: 200+ correct trajectories
  └── Rebuild SFT dataset with build_sft_dataset.py

Phase B: SFT Training (AdamW baseline)
  ├── Verify model ID matches inference model
  ├── Split dataset: 85% train, 15% val
  ├── Train with AdamW, cosine schedule, TensorBoard logging
  ├── Save checkpoints every 25 steps
  ├── Monitor: training loss, val loss, gradient norms
  ├── Early stop if val loss plateaus
  └── Eval: run NIAH suite on trained adapter

Phase C: SFT Training (Muon ablation)
  ├── Fix Muon momentum warmup
  ├── Fix embedding exclusion in split_params_for_muon()
  ├── Fix Muon LR for LoRA (start at 0.005)
  ├── Train with same data/config, only optimizer different
  ├── Compare: convergence speed, final loss, NIAH accuracy
  └── Plot: AdamW vs Muon training curves side-by-side

Phase D: Analysis & Publication
  ├── Merge best LoRA adapter into base model
  ├── Run full benchmark suite on merged model
  ├── Compare: base model vs SFT-AdamW vs SFT-Muon
  ├── Plot: accuracy curves, training curves, cost comparison
  └── Write up findings
```

---

## Priority Order

1. **Re-run benchmarks** with fixed RLM runtime → collect clean trajectories (BLOCKS everything else)
2. **Fix Muon implementation** (momentum warmup, embedding exclusion, LR adjustment)
3. **Fix _CombinedOptimizer** or replace with proper hook
4. **Add TensorBoard logging** + validation split + gradient clipping
5. **Add training curves plotting script**
6. **Resolve model ID mismatch**
7. **Run actual training** when dataset is large enough
