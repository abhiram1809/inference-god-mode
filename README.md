# Inference God Mode

A reusable coding-agent skill for planning, deploying, benchmarking, and optimizing self-hosted open-weight LLM inference, from one GPU to Kubernetes clusters.

Start with [the inference-optimizer skill](.codex/skills/inference-optimizer/SKILL.md).

## What it covers

- Hardware inventory, runtime-specific memory sizing, full native context, and realistic concurrency.
- Live checkpoint discovery, 4-bit weight candidates, NVFP4 kernel verification, and FP8 KV evaluation.
- vLLM, SGLang, TensorRT-LLM, and llama.cpp; MLX-LM/vLLM-Metal for Apple Silicon. Ollama is excluded.
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages contract testing.
- MTP and other speculative decoding methods, prefix reuse, and CPU/SSD KV tiers.
- Community builds and quantization plugins, issue-first troubleshooting, and legacy GPU/Vulkan routing.
- Sequential multi-GPU TP, PP, DP, and MoE EP experiments, with realistic serving-load comparisons.
- llm-d cache-aware routing, prefill/decode disaggregation, pool sizing, and production workload benchmarks.

Defaults are one active user when the workload is vague, a 4-bit weight candidate, and the model's full native context. FP8 KV is the preferred optimized candidate where supported and validated. The skill asks for requirements that affect deployment and keeps correctness, quality, memory, and latency checks in the optimization loop.

## Use the skill

Clone the repository:

```bash
git clone https://github.com/abhiram1809/inference-god-mode.git
```

Point your coding harness at `.codex/skills/inference-optimizer/SKILL.md` and ask it to use the skill for your model and hardware. For another project, copy the complete `inference-optimizer` folder into the skill directory supported by your harness; keep `references/` beside `SKILL.md`.

This repository provides these project paths:

| Harness | Skill path |
| --- | --- |
| Codex | `.codex/skills/inference-optimizer/` |
| Claude Code | `.claude/skills/inference-optimizer/` |
| OpenCode | `.opencode/skills/inference-optimizer/` |

The Claude Code and OpenCode paths are relative symlinks to the canonical Codex folder. Preserve symlinks when checking out the repository, or copy the canonical folder into those paths if your platform does not restore them.

Example request:

```text
Use inference-optimizer to serve <model repository> on <GPU and memory>.
I need <API dialects>, <peak active generations>, and <latency target>.
Keep the full native context and evaluate 4-bit weights and FP8 KV.
Benchmark the actual workload and report the best verified configuration.
```

## Evidence and scope

The skill requires current upstream documentation, exact checkpoint and build revisions, and measurements on the target workload. Routing preferences and community recipes are starting candidates. Follow the linked references for compatibility checks, quality controls, and benchmark procedures.

Published validation covers skill structure, local reference links, and harness symlinks. Deployment performance must be measured for the chosen model, hardware, and runtime.
