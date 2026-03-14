# Implementation Plan

## Overview
We will compare Recursive Language Models (RLMs) against standard LLM inference using one fixed local model: `qwen3.5:2b` via Ollama. The goal is to isolate architecture differences (vanilla vs recursive) without provider/model variance.

## Goal Description
Build a pipeline evaluating context-rot behavior. We will input long prompts to the base model and compare its accuracy against the same model structured to recursively reason via a Python REPL loop.

## Proposed Changes
### Core System
- [NEW] `rlm_core.py` (or similar): main REPL execution loop and recursion control.
- [NEW] `evaluation.py`: assess vanilla accuracy vs RLM accuracy on the same tasks.

### Workflows
- Phase 1: Setup minimal infrastructure for REPL and Ollama model communication.
- Phase 2: Establish vanilla baseline with `qwen3.5:2b`.
- Phase 3: Develop RLM execution loop over the same model.
- Phase 4: Run comparisons, evaluate datasets, log metrics.

## Verification Plan
1. **Automated Tests:** scripts measuring accuracy on known long-context tasks.
2. **Resource Metrics:** memory and runtime must stay stable on RTX 4050 6GB.
3. **Comparability:** both methods must use the same model and prompt family.
