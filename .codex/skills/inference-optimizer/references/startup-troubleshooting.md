# Issue-first startup and inference troubleshooting

Read when model loading, kernel compilation, server startup, first generation, or ongoing inference fails or hangs. Search primary GitHub issues before speculative fixes; use the harness to collect evidence and apply a justified correction. An LLM-generated explanation alone is insufficient grounds to rewrite the serving arguments.

## Capture and search

Preserve the exact launch command and environment, model/checkpoint revision, engine/fork/plugin and image identity, GPU/compute capability, driver/runtime, attention and quantization backend, and complete relevant traceback. Find the earliest substantive worker error; a final “engine failed to start” wrapper is usually a poor search key. For a hang, record the last completed stage and use documented logging to identify the stalled operation. Remove credentials and private prompt data from search queries and shared logs.

Search both open and closed issues in the engine repository using distinctive error text, failing symbol/operator, GPU architecture, model, and version. For example, `repo:vllm-project/vllm is:issue "<distinctive error>"`. Broaden the query if exact matching fails. Follow dependency errors to the responsible repository (attention library, compiler, PyTorch, driver integration, or quantization plugin). For a community build, search both its fork/plugin tracker and upstream; an upstream fix may be absent from the image.

Primary trackers: [vLLM](https://github.com/vllm-project/vllm/issues), [SGLang](https://github.com/sgl-project/sglang/issues), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/issues), [llama.cpp](https://github.com/ggml-org/llama.cpp/issues).

For cluster failures, distinguish engine errors from Kubernetes scheduling, gateway/Endpoint Picker routing, and KV-transfer errors. Follow [llm-d's component repositories](https://github.com/llm-d) to the responsible issue tracker; capture pod events and component logs alongside the engine traceback.

## Resolve without losing the target

- Read the issue discussion and linked PR/commit, not just the title or first workaround. Prefer maintainer-confirmed causes and reproductions matching the installed model, GPU, and dependency versions. Check whether a fix actually merged and which release/image contains it. An old closed issue or a similar error is a lead, not proof of the same cause.
- Verify proposed flags against the installed CLI/docs or source. Apply the smallest evidence-backed change, record the issue/commit and rationale, and keep the previous environment recoverable. Avoid bundles of guessed environment variables, unsupported architecture overrides, and blind dependency upgrades.
- A reduced context/batch, eager execution, disabled feature, or dummy load can isolate a failure. Label it as a **temporary diagnostic**, record the command delta, and restore the required settings before acceptance. Do not claim success after silently dropping full context, quantization, speculation, concurrency, modalities, or API features. If a required feature is genuinely unsupported, explain the evidence and evaluate a supported route within the user's constraints.
- Restart from a clean diagnostic environment, remove debug/profiling overrides, then verify startup, a real generation, full-context/concurrency capacity, quality, and required APIs. Rerun affected performance measurements when the fix changes kernels or dependencies. Report the working command and source evidence.

If no matching issue exists, use current upstream troubleshooting docs and inspect the failing code; state which hypotheses remain unverified. Produce a minimal reproduction and environment report for further investigation. If browsing is unavailable, report that evidence gap instead of inventing an issue or fix. Stop repeating retries when no new evidence is obtained.

Reference: [vLLM troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/) recommends searching existing issues, tracing root causes, and removing diagnostic settings after debugging.
