# Hardware Reality Check: Single-Model Local Strategy (RTX 4050 6GB VRAM)

---

## Our Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4050 6GB VRAM |
| Architecture | Ada Lovelace |
| VRAM | 6GB |
| System RAM | 16-32GB (assumed) |

---

## Hard Truth: What 6GB VRAM Can and Cannot Do

### Cannot Do
- Run 70B-class models locally at useful speed/quality
- Large-scale post-training of 7B+ models locally
- Extremely large-context real-time inference with high throughput

### Can Do
- Run compact local models through Ollama
- Build and benchmark an RLM runtime wrapper
- Compare vanilla vs RLM behavior on the same local model
- Run controlled experiments with clear reproducibility

---

## Locked Strategy (No API Models)

> **Decision:** We use only one model for all experiments.

```
Model runtime: Ollama (local)
Model:         qwen3.5:2b
Root LM:       qwen3.5:2b
Sub LM:        qwen3.5:2b
Comparison:    Vanilla qwen3.5:2b vs RLM-wrapped qwen3.5:2b
Budget:        Local-only execution
```

Why this is better for our project:
- Removes provider variance and API instability
- Improves reproducibility and comparability
- Keeps architecture comparison clean (same model everywhere)

---

## Practical Constraints for This Setup

- RLM still needs chunking and iterative decomposition; sub-calls must fit the active context window.
- Latency may be high for many recursive calls on local hardware.
- Benchmark scale should be increased gradually (start small, then scale up).

---

## Summary: What We Will Build (Locked In)

| Component | Choice |
|---|---|
| Package manager | `uv` |
| Language | Python 3.12 |
| REPL type | Local exec-based (LocalREPL) |
| Model runtime | Ollama |
| Base model | `qwen3.5:2b` |
| Root LM | `qwen3.5:2b` |
| Sub LM | `qwen3.5:2b` |
| Baseline LLM | Same model, direct context (no REPL) |
| Benchmarks | NIAH + custom long-doc QA (expand later) |
| Primary goal | Fair architecture comparison with one fixed model |
