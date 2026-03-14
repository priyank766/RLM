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
    uv run python scripts/train_sft.py --optimizer adamw --epochs 3
    uv run python scripts/train_sft.py --lora-rank 16 --lr 2e-4
    uv run python scripts/train_sft.py --resume experiments/checkpoints/sft_adamw/checkpoint-50
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
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from training.muon import Muon, split_params_for_muon


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-1.5B"  # HF model matching qwen3.5:2b capabilities
# Note: Qwen3.5-2B may not be on HF yet. Using Qwen2.5-1.5B as fallback.
# Change to "Qwen/Qwen3.5-2B" when available.

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]


def load_sft_dataset(path: str) -> Dataset:
    """Load JSONL SFT dataset into HuggingFace Dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            records.append(item)

    # SFTTrainer expects a "messages" column for chat template training
    return Dataset.from_list(records)


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


def create_adamw_trainer(model, tokenizer, dataset, args, output_dir):
    """Standard AdamW SFTTrainer."""
    training_args = SFTConfig(
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
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        optim="adamw_torch",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    return trainer


def create_muon_trainer(model, tokenizer, dataset, args, output_dir):
    """
    SFTTrainer with Muon optimizer for 2D weights + AdamW for the rest.

    Strategy: We use a custom training loop via TRL's SFTTrainer but
    override the optimizer creation.
    """
    training_args = SFTConfig(
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
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Override optimizer with Muon + AdamW split
    muon_params, adamw_params = split_params_for_muon(model)

    print(f"\nMuon optimizer split:")
    print(f"  2D params (Muon):  {len(muon_params)} tensors, "
          f"{sum(p.numel() for p in muon_params):,} parameters")
    print(f"  Other (AdamW):     {len(adamw_params)} tensors, "
          f"{sum(p.numel() for p in adamw_params):,} parameters")

    muon_opt = Muon(muon_params, lr=args.muon_lr, momentum=0.95)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=0.01)

    # Combine into a single optimizer group via a wrapper
    trainer.optimizer = _CombinedOptimizer(muon_opt, adamw_opt)

    return trainer


class _CombinedOptimizer(torch.optim.Optimizer):
    """Wraps two optimizers into one interface for HF Trainer."""

    def __init__(self, opt1, opt2):
        # We don't call super().__init__ — this is a shim
        self.opt1 = opt1
        self.opt2 = opt2
        self.defaults = {}
        self.state = {}
        self.param_groups = opt1.param_groups + opt2.param_groups

    def step(self, closure=None):
        self.opt1.step(closure)
        self.opt2.step(closure)

    def zero_grad(self, set_to_none=True):
        self.opt1.zero_grad(set_to_none=set_to_none)
        self.opt2.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"opt1": self.opt1.state_dict(), "opt2": self.opt2.state_dict()}

    def load_state_dict(self, state_dict):
        self.opt1.load_state_dict(state_dict["opt1"])
        self.opt2.load_state_dict(state_dict["opt2"])


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (keep 1 for 6GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (AdamW)")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Muon learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (lower = less memory)")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
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
    dataset = load_sft_dataset(args.dataset)
    print(f"Dataset: {len(dataset)} examples")

    # Load model
    model, tokenizer = build_model_and_tokenizer(args.model, args.lora_rank, args.lora_alpha)

    # Create trainer
    output_dir = f"experiments/checkpoints/sft_{args.optimizer}"
    if args.optimizer == "muon":
        trainer = create_muon_trainer(model, tokenizer, dataset, args, output_dir)
    else:
        trainer = create_adamw_trainer(model, tokenizer, dataset, args, output_dir)

    # Train
    print(f"\nStarting training -> {output_dir}")
    print(f"{'='*60}")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete! Model saved to {final_dir}")

    # Log training metrics
    metrics = trainer.state.log_history
    metrics_path = Path(output_dir) / "training_metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    print(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
