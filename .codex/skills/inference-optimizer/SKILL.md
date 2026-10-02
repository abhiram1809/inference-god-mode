---
name: inference-optimizer
description: Plan, deploy, benchmark, and tune self-hosted open-weight LLM inference on one machine or Kubernetes GPU clusters, including quantization, full-context capacity, OpenAI or Anthropic APIs, cache-aware routing, and prefill/decode disaggregation.
---

# Inference optimizer

Optimize the user's *specific* model, hardware, workload, and client protocol. Work from a measured baseline to a verified best configuration; do not present an untested flag list as an optimization result.

## Establish the target

Collect or inspect these facts, then ask only for missing decisions that affect the route:

- Exact model repository and revision, architecture, license/gating, tokenizer and chat template, modalities, reasoning and tool-use requirements.
- GPU model, count, memory capacity and bandwidth per GPU, compute capability, interconnect/topology, driver/runtime, host RAM and bandwidth, storage, operating system, and whether the machine is shared. Use read-only hardware commands where available rather than asking for facts the machine can report.
- Peak simultaneous *active generations*, arrival rate or requests per minute, typical and worst-case input/output lengths, latency target (TTFT and inter-token latency), and throughput target. Number of registered users is not concurrency. Ask for the minimum context window the user would accept **if** the full native window proves infeasible.
- Required wire protocols and client features: OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, streaming, tools, structured output, reasoning fields, image/audio support, token counting, and authentication.
- Quality tolerance, time and compute budget for local quantization, and whether weights may be downloaded or published. Ask whether preserving reusable KV prefixes in host RAM or SSD is desirable when the workload has repeated prefixes, and capture the available RAM/SSD budget.

If workload is vague, provisionally assume one active user and prioritize interactive latency. Default to a **4-bit weight candidate** unless the user chooses another precision; this is a search starting point, not a claim that 4-bit is fastest or accurate enough. Assume the user wants the model's full *native* context window, including output-token headroom. Never silently shorten context or apply RoPE scaling to claim a larger window. If full context cannot fit at the requested concurrency, calculate the gap and ask which constraint may change: concurrency, GPU capacity, KV precision, model, or context. Continue independent research while waiting.

Prefer **FP8 KV cache** as the first optimized candidate where the model, device, and attention backend support it. Keep a BF16/FP16 KV control and promote FP8 after model-specific quality, full-context, and performance checks. Weight precision and KV precision are independent. Read [FP8 KV selection](references/fp8-kv-cache.md) for kernel fixes, scales, and exceptions.

## Select candidates

Read [backend routing](references/backend-routing.md) for the relevant device and [quantization and capacity](references/quantization-capacity.md) before choosing weights. Refresh upstream compatibility documentation, release notes, and the live Hugging Face Hub for the exact model; version-specific support and checkpoint quality change often.

When upstream lacks a required model/GPU/quantization combination, search official model recipes, merged changes/nightlies, plugins, and community images or forks. Read [community builds](references/community-builds.md) before recommending or running one; pin its source and image, establish what it changes, and verify its actual kernel path. An EXL3 plugin can be evaluated within vLLM without changing the allowed engine set.

- Legacy GPUs: evaluate **llama.cpp** first when current production engines or their required kernels no longer support the device. For older AMD cards such as Radeon RX 560X, prefer a **Vulkan** candidate; verify the exact GPU and driver capabilities, actual offload, and workload fit. Apply this compatibility route before the workload defaults below. Read the legacy section of [backend routing](references/backend-routing.md).
- Supported NVIDIA GPUs, multiple concurrent users: start with **vLLM**. Supported NVIDIA GPUs, one active user: start with **SGLang**, while keeping vLLM as the comparator. A hard requirement for all three API dialects may favor a version of vLLM that implements and passes each required route. These are starting hypotheses; benchmark alternatives on the actual workload.
- Consider **TensorRT-LLM** when the model architecture, checkpoint, GPU, and current serving API are supported and an engine build is justified by measured gains. Consider **llama.cpp** for GGUF checkpoints, constrained memory, unusual hardware, or a measured single-user advantage. Do not route to Ollama.
- Apple Silicon: consider **MLX-LM** or **vLLM-Metal** with verified MLX 4-bit checkpoints; consider llama.cpp/GGUF if API coverage or model support calls for it. Do not assume MLX-LM's basic HTTP server supplies all requested API dialects.
- A checkpoint's label is insufficient: verify its quantization metadata, architecture, tokenizer, license, producer, calibration method where available, backend loader, and **actual launched kernel and arithmetic**. On Blackwell, evaluate NVFP4 first **when a genuine compatible checkpoint and kernel exist**. On Ampere/Ada/Hopper, evaluate AWQ/GPTQ with the available optimized kernels, plus other supported 4-bit candidates. NF4 is not interchangeable with AWQ/GPTQ and does not imply fast serving. Distinguish 4-bit storage from 4-bit activations and math; a successful load does not prove a native FP4 path.

Treat each required endpoint and behavior as a tested contract. If the preferred engine lacks an endpoint, use another allowed engine or a clearly identified adapter only if that adapter passes end-to-end tests; never describe partial compatibility as full compatibility. Read [API verification](references/api-verification.md).

## Execute the optimization loop

Read [optimization loop](references/optimization-loop.md) for sizing, benchmark design, tuning order, and kernel work. In brief:

1. Inventory hardware and exact model. Search all plausible Hugging Face quantized variants; shortlist by actual format, backend support, provenance, and fit. Pin repository revisions and engine versions.
2. Estimate weight, KV-cache, runtime, graph/workspace, and concurrent-request memory **for the chosen engine and kernel path** at the full native context. A GGUF file size or llama.cpp memory result does not establish vLLM/SGLang/TensorRT-LLM fit. Verify with a real load and full-context request; retain OOM headroom.
3. Run the simplest correct configuration. Exercise every required API feature and a small quality set against an appropriate higher-precision reference.
4. Benchmark representative short, medium, full-context, single-stream, and target-concurrency cases. Record TTFT, inter-token latency, tokens/s, p50/p95 and sufficiently sampled p99 latency, error rate, peak memory, power if relevant, versions, and exact commands.
5. Change one performance dimension at a time: weight/kernel choice, FP8 KV, scheduling/batching and CPU/GPU overlap, prefill, graph capture, prefix caching, parallelism, then [speculative decoding and KV offload](references/speculation-kv-offload.md) where applicable. Search for a compatible native or attached MTP head, EAGLE/other drafter, or DFlash checkpoint for the exact model and engine. Consider CPU/SSD KV tiers only for measured reusable-prefix benefit. Retest correctness and quality after each change. Keep the best Pareto choices for latency, throughput, memory, and quality.
6. If no suitable 4-bit checkpoint exists, propose or execute a local quantization path with calibration and evaluation. If profiling identifies a real kernel bottleneck and the user wants further engineering, use a pinned fresh upstream checkout, make a narrow reproducible change, and compare against the baseline. Do not claim a speedup from synthetic microbenchmarks alone.

For multi-GPU setups, read [parallelism experiments](references/parallelism-experiments.md) and evaluate supported layouts **sequentially: TP → PP → DP → EP**, with EP only for supported MoE models. Inspect dense versus MoE architecture, per-GPU fit, topology, and valid combinations before generating commands. Present the experiment matrix and ask for missing run-budget or service-interruption constraints; honor existing authorization. Keep a TP=1 control when it fits, but do not settle on it without evaluating useful multi-GPU layouts. Measure real serving traffic before selecting the final configuration; a canned throughput score or successful startup is preliminary evidence.

Do not download large checkpoints, run long calibration, rebuild engines, expose a service beyond localhost, or publish weights as an incidental step. Make those costs and side effects explicit before taking them on; follow the user's existing authorization.

## Organizational and multi-machine deployment

For organizational fleet serving or Kubernetes deployment, read [cluster serving](references/cluster-serving.md) after establishing a correct engine baseline. Evaluate **llm-d** as an orchestration and routing layer above the allowed engines. Compare replicas with ordinary routing, cache/load-aware routing, and then separate prefill/decode pools when measured interference justifies them. Size both stages from the workload and network measurements; preserve full-context, API, quality, and availability requirements through the complete gateway path. Prepare reproducible manifests or Helm values and an end-to-end benchmark before claiming a production configuration is optimized.

## When startup or inference fails

Read [issue-first troubleshooting](references/startup-troubleshooting.md). Capture the root error and exact environment, then search matching upstream GitHub issues and their linked fixes before trying speculative argument changes. Prefer maintainer explanations, reproducible reports, and current code/docs over an LLM's unsupported diagnosis. Preserve the intended model, quantization, context, concurrency, and API contract; distinguish temporary diagnostic settings from the final configuration and retest after fixing the cause.

## Deliverable

Give the user a runnable configuration or a concrete next experiment, exact model/checkpoint/revision, backend/version, image digest and fork/plugin commits if used, kernel libraries and selected paths, tested API routes, full-context and concurrency limits, benchmark table, quality observations, and remaining bottleneck. For multi-GPU runs, include the tested TP/PP/DP/EP layouts, rank placement, per-GPU memory, skipped/failed candidates, and the reason for choosing the winner. Mark estimates and untested suggestions as such. If no configuration meets the hard constraints, show the binding constraint and smallest viable alternatives.
