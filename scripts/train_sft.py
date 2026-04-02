"""
SFT training for RLM trajectories using QLoRA.

Supports two optimizer modes for ablation:
  --optimizer adamw   (baseline)
  --optimizer muon    (Muon for 2D matrices + AdamW for embeddings/layernorm)

Prerequisites:
  1. Stop Ollama to free VRAM:  ollama stop qwen3.5:2b
  2. Install training deps:     uv pip install -e ".[train]"
  3. Build SFT dataset:         uv run python scripts/build_sft_dataset.py
  4. Run training:              uv run python scripts/train_sft.py

Usage:
    uv run python scripts/train_sft.py
    uv run python scripts/train_sft.py --optimizer muon
    uv run python scripts/train_sft.py --optimizer adamw --epochs 5
    uv run python scripts/train_sft.py --lora-rank 16 --lr 2e-4
    uv run python scripts/train_sft.py --resume experiments/checkpoints/sft_adamw/checkpoint-50

Changes (2026-03-20):
    - Fixed CombinedOptimizer to work properly with HF Trainer
    - Added TensorBoard logging
    - Added validation split (15%)
    - Added gradient clipping (max_grad_norm=1.0)
    - Updated hyperparameter defaults (lora_rank=16, alpha=32)
    - Added training curves plot generation at end
    - Added model merge script at end of training
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure CUDA memory is efficient
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from training.muon import Muon, split_params_for_muon


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B"  # HF model for training
# NOTE: This should match the model used in Ollama for inference.
# If you're running qwen3.5:2b in Ollama, check if Qwen/Qwen3-2B is
# available on HuggingFace and switch to that for consistency.

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]


def load_sft_dataset(path: str, val_split: float = 0.15) -> tuple[Dataset, Dataset | None]:
    """Load JSONL SFT dataset into HuggingFace Dataset with optional val split."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(item)

    if not records:
        print("ERROR: Dataset is empty")
        sys.exit(1)

    dataset = Dataset.from_list(records)

    # Validation split only if we have enough data
    if val_split > 0 and len(records) >= 10:
        split = dataset.train_test_split(test_size=val_split, seed=42)
        print(f"  Train: {len(split['train'])} examples, Val: {len(split['test'])} examples")
        return split["train"], split["test"]
    else:
        print(f"  Train: {len(dataset)} examples (no val split — too few examples)")
        return dataset, None


def build_model_and_tokenizer(model_id: str, lora_rank: int, lora_alpha: int):
    """Load 4-bit quantized model with LoRA adapters."""

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # nested quantization saves more memory
    )

    print(f"Loading model: {model_id} (4-bit QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def _make_training_args(args, output_dir: str, has_eval: bool) -> SFTConfig:
    """Build SFTConfig shared by both optimizers."""
    eval_kwargs = {}
    if has_eval:
        eval_kwargs = {
            "eval_strategy": "steps",
            "eval_steps": max(1, args.save_steps),
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        max_grad_norm=1.0,         # gradient clipping
        optim="adamw_torch",
        report_to="tensorboard",   # TensorBoard logging
        logging_dir=f"{output_dir}/logs",
        seed=42,
        **eval_kwargs,
    )


def create_adamw_trainer(model, tokenizer, train_ds, eval_ds, args, output_dir):
    """Standard AdamW SFTTrainer."""
    training_args = _make_training_args(args, output_dir, has_eval=eval_ds is not None)

    callbacks = []
    if eval_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=callbacks,
    )
    return trainer


def create_muon_trainer(model, tokenizer, train_ds, eval_ds, args, output_dir):
    """
    SFTTrainer with Muon optimizer for 2D weights + AdamW for the rest.

    Strategy: We create the trainer with AdamW, then override the optimizer
    with a CombinedOptimizer that wraps Muon + AdamW.
    """
    training_args = _make_training_args(args, output_dir, has_eval=eval_ds is not None)

    callbacks = []
    if eval_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=callbacks,
    )

    # Override optimizer with Muon + AdamW split
    muon_params, adamw_params = split_params_for_muon(model)

    print(f"\nMuon optimizer split:")
    print(f"  2D params (Muon):  {len(muon_params)} tensors, "
          f"{sum(p.numel() for p in muon_params):,} parameters")
    print(f"  Other (AdamW):     {len(adamw_params)} tensors, "
          f"{sum(p.numel() for p in adamw_params):,} parameters")

    muon_opt = Muon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=0.01)

    # Combine into a single optimizer group via a wrapper
    trainer.optimizer = _CombinedOptimizer(muon_opt, adamw_opt)

    return trainer


class _CombinedOptimizer(torch.optim.Optimizer):
    """
    Wraps two optimizers into one interface for HF Trainer.

    This properly delegates all Optimizer methods to both sub-optimizers
    and maintains a unified param_groups list for the Trainer's LR scheduler.
    """

    def __init__(self, opt1, opt2):
        # We don't call super().__init__() — this is a shim
        self.opt1 = opt1
        self.opt2 = opt2
        self.defaults = {}
        # Unified param_groups so HF Trainer's LR scheduler can iterate
        self.param_groups = opt1.param_groups + opt2.param_groups

    @property
    def state(self):
        """Merge state dicts from both optimizers."""
        merged = {}
        merged.update(self.opt1.state)
        merged.update(self.opt2.state)
        return merged

    def step(self, closure=None):
        loss1 = self.opt1.step(closure)
        loss2 = self.opt2.step(None)  # Don't re-run closure
        return loss1 if loss1 is not None else loss2

    def zero_grad(self, set_to_none=True):
        self.opt1.zero_grad(set_to_none=set_to_none)
        self.opt2.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "opt1": self.opt1.state_dict(),
            "opt2": self.opt2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.opt1.load_state_dict(state_dict["opt1"])
        self.opt2.load_state_dict(state_dict["opt2"])

    def add_param_group(self, param_group):
        """Route to opt2 (AdamW) by default."""
        self.opt2.add_param_group(param_group)
        self.param_groups = self.opt1.param_groups + self.opt2.param_groups


# ---------------------------------------------------------------------------
# Post-training utilities
# ---------------------------------------------------------------------------

def plot_training_curves(metrics_path: Path, output_dir: Path):
    """Generate training loss curve from saved metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping training curve plot")
        return

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if "loss" in entry and "step" in entry:
                train_steps.append(entry["step"])
                train_loss.append(entry["loss"])
            if "eval_loss" in entry and "step" in entry:
                eval_steps.append(entry["step"])
                eval_loss.append(entry["eval_loss"])

    if not train_steps:
        print("No training loss data found — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_steps, train_loss, label="Train Loss", alpha=0.7, linewidth=1.5)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="Val Loss", alpha=0.9, linewidth=2,
                marker="o", markersize=4)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training Curves — {output_dir.name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {plot_path}")


def merge_adapter(adapter_dir: Path, base_model_id: str, output_dir: Path):
    """Merge LoRA adapter into base model for deployment."""
    from peft import PeftModel

    print(f"\nMerging adapter from {adapter_dir} into {base_model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()

    merged_dir = output_dir / "merged"
    merged.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT training with QLoRA + Muon/AdamW")
    parser.add_argument("--dataset", default="experiments/sft_dataset.jsonl",
                        help="SFT dataset JSONL path")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                        help="Optimizer: adamw (baseline) or muon (ablation)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (keep 1 for 6GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (AdamW)")
    parser.add_argument("--muon-lr", type=float, default=0.005, help="Muon learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (2*rank)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (lower = less memory)")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data for validation (0 = no split)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA adapter after training")
    args = parser.parse_args()

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run: uv run python scripts/build_sft_dataset.py")
        sys.exit(1)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Need GPU for training.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
    mem_free = (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()) / 1024**3
    print(f"GPU: {gpu} ({mem_total:.1f}GB total, ~{mem_free:.1f}GB free)")
    print(f"Optimizer: {args.optimizer}")
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print()

    # Load dataset
    train_ds, eval_ds = load_sft_dataset(args.dataset, val_split=args.val_split)
    print(f"Dataset total: {len(train_ds) + (len(eval_ds) if eval_ds else 0)} examples")

    # Load model
    model, tokenizer = build_model_and_tokenizer(args.model, args.lora_rank, args.lora_alpha)

    # Create trainer
    output_dir_path = Path(f"experiments/checkpoints/sft_{args.optimizer}")
    output_dir = str(output_dir_path)

    if args.optimizer == "muon":
        trainer = create_muon_trainer(model, tokenizer, train_ds, eval_ds, args, output_dir)
    else:
        trainer = create_adamw_trainer(model, tokenizer, train_ds, eval_ds, args, output_dir)

    # Train
    print(f"\nStarting training -> {output_dir}")
    print(f"TensorBoard: tensorboard --logdir={output_dir}/logs")
    print(f"{'='*60}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete! Adapter saved to {final_dir}")

    # Log training metrics
    metrics = trainer.state.log_history
    metrics_path = output_dir_path / "training_metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    print(f"Training metrics saved to {metrics_path}")

    # Plot training curves
    plot_training_curves(metrics_path, output_dir_path)

    # Optional: merge adapter
    if args.merge:
        merge_adapter(Path(final_dir), args.model, output_dir_path)


if __name__ == "__main__":
    main()
