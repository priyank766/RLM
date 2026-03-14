"""
Evaluate SFT-trained LoRA model vs base model on NIAH benchmark.

Loads the QLoRA adapter, merges with base, and runs the same NIAH suite
through the RLM pipeline. Compares against vanilla and base-RLM results.

Usage:
    uv run python scripts/eval_sft.py experiments/checkpoints/sft_adamw/final
    uv run python scripts/eval_sft.py experiments/checkpoints/sft_muon/final --lengths 4000,8000,16000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.niah import generate_niah_suite
from comparison.eval import score_result


class HFClient:
    """LLM client using a HuggingFace model (for eval of SFT models)."""

    def __init__(self, model, tokenizer, max_new_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def complete(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_sft_model(adapter_path: str, base_model_id: str):
    """Load base model + LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def evaluate_model(client, tasks, label: str) -> list[dict]:
    """Run RLM eval on a list of NIAH tasks."""
    from rlm.rlm_repl import RLM_REPL

    rlm = RLM_REPL(root_client=client, sub_client=client, max_iterations=20)
    results = []

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {label} ctx={task.context_length:,}ch ...", end=" ", flush=True)
        t0 = time.time()

        result = rlm.completion(task.context, task.query)
        lat = time.time() - t0

        prediction = str(result["answer"] or "")
        scores = score_result(prediction, task.answer, task.query)
        correct = scores.get("contains_answer", False)

        status = "PASS" if correct else ("TIMEOUT" if result["timed_out"] else "FAIL")
        print(f"{status} lat={lat:.1f}s iters={result['iterations']}")

        results.append({
            "method": label,
            "task_type": "niah",
            "context_length": task.context_length,
            "query": task.query,
            "answer": task.answer,
            "prediction": prediction,
            "correct": correct,
            "scores": scores,
            "latency_s": round(lat, 2),
            "timed_out": result["timed_out"],
            "iterations": result["iterations"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model on NIAH")
    parser.add_argument("adapter_path", help="Path to LoRA adapter directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B",
                        help="Base model ID")
    parser.add_argument("--lengths", default="4000,8000,16000",
                        help="Context lengths to test")
    parser.add_argument("--out", default=None, help="Output JSONL path")
    args = parser.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    positions = [0.1, 0.5, 0.9]
    tasks = generate_niah_suite(context_lengths=lengths, positions=positions)

    print(f"NIAH eval: {len(tasks)} tasks, lengths={lengths}")

    # Load model
    model, tokenizer = load_sft_model(args.adapter_path, args.base_model)
    client = HFClient(model, tokenizer)

    # Run eval
    adapter_name = Path(args.adapter_path).parent.name
    results = evaluate_model(client, tasks, label=f"sft_{adapter_name}")

    # Summary
    correct = sum(1 for r in results if r["correct"])
    print(f"\nResults: {correct}/{len(results)} correct ({100*correct/len(results):.0f}%)")

    # Save
    out_path = args.out or f"experiments/eval_{adapter_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
