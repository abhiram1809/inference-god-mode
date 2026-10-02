# Sequential multi-GPU parallelism experiments

Read when multiple GPUs are available on one machine or across nodes. Evaluate **TP, then PP, then DP, then EP** as families of feasible layouts. They can compose; EP applies to MoE expert layers. Keep a single-GPU control if the full workload fits, and select the winner using the user's latency/throughput objective.

## Prepare a reviewable experiment matrix

Inspect dense versus MoE architecture, total and active parameters, attention/KV scheme, layer/head/expert dimensions, checkpoint precision, and model-specific parallel support. Inventory per-GPU memory, heterogeneous devices, PCIe/NVLink/NVSwitch topology, cross-node transport, and actual rank placement. Reuse [runtime capacity estimates](quantization-capacity.md); pooled memory is insufficient evidence that every rank fits. Confirm support for the required APIs, full context, quantized kernels, graphs, and speculation at each layout.

List viable degrees and combinations, exact commands, GPU assignments, representative workload, expected startup/run duration, cost ceiling, and whether experimentation interrupts an existing service. Ask once for missing time/resource/maintenance-window constraints before costly runs or restarts. Existing authorization covers experiments within the agreed scope. Continue inventory and command preparation while waiting on a required decision; do not treat elapsed time as permission.

Fix model revision, weights, KV dtype, context, sampling, prompt/output distribution, cache policy, arrival trace, and engine build across the initial comparison. Keep the GPU allocation/cost budget comparable and report active GPU counts. Use one candidate at a time: launch, verify, warm, measure, save results, and shut down its workers before the next candidate. Concurrent contenders on the same hardware contaminate measurements. Restart from a recorded configuration rather than accumulating flags.

## Evaluate the families in order

1. **Tensor parallelism (TP):** Try supported TP degrees with PP=1 and ordinary DP=1. Include TP=1 where the weights, full-context KV, and target concurrency fit; otherwise begin at a feasible degree. Check sharding constraints and measure collective time and per-rank KV/workspace. Larger TP can help fit and throughput while adding communication cost.
2. **Pipeline parallelism (PP):** Compare feasible PP-only and TP+PP layouts on the same resources. Test weaker-interconnect and uneven-partition cases where supported. Inspect stage imbalance, pipeline bubbles, activation transfers, and single-stream latency. After the initial layout comparison, tune microbatch/prefill chunking within each promising PP layout. PP is not prefill/decode disaggregation. See [vLLM scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) and [SGLang PP](https://docs.sglang.io/docs/advanced_features/pipeline_parallelism).
3. **Data parallelism (DP):** Test supported replica layouts with enough model/KV capacity per replica, combining TP/PP for fit when necessary. Distribute requests across replicas using the intended routing policy and include routing/cache effects in the measurement. For dense models, replicas increase aggregate serving capacity; they do not divide one request's model computation. MoE attention-DP can share distributed expert layers and has different memory/group semantics. Verify the engine's DP versus attention-DP behavior rather than copying degrees between engines. See [vLLM DP](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) and [SGLang DP/DPA](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide).
4. **Expert parallelism (EP), MoE only:** Compare a valid TP/DP base with expert sharding enabled where supported; account for necessary topology changes. Verify expert placement, all-to-all dispatch/combine, quantized grouped kernels, graph support, and any additional communication dependencies. Measure expert imbalance and communication cost; tune the supported all-to-all backend or expert load balancing only after a correct EP baseline, accounting for redundant-expert memory. Skip this family for dense models. See [vLLM EP](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) and [SGLang EP](https://docs.sglang.io/docs/advanced_features/expert_parallelism).

For four GPUs, a **candidate** dense-model grid could compare `TP=4, PP=1, DP=1`; `TP=2, PP=2, DP=1`; and two replicas with `TP=2, PP=1`. Pure PP=4 or four TP=1 replicas are additional candidates when supported and when each rank/replica fits. These are layouts to qualify, not generic launch prescriptions. Skip an unsupported/infeasible case with its evidence; do not silently shorten context or remove a required feature to get a benchmark number.

## Translate layouts into the installed engine's arguments

Inspect current CLI help, release-specific docs, and startup rank/group logs. These documented argument names are lookup starting points:

| Dimension | vLLM | SGLang |
| --- | --- | --- |
| TP | `--tensor-parallel-size` | `--tp-size` / `--tensor-parallel-size` |
| PP | `--pipeline-parallel-size` | `--pp-size` / `--pipeline-parallel-size` |
| Ordinary DP | `--data-parallel-size`, supported internal/external routing mode | `--dp-size` / `--data-parallel-size`, supported replica routing |
| EP | `--enable-expert-parallel`, supported `--all2all-backend` | `--ep-size` / `--expert-parallel-size`, supported `--moe-a2a-backend` |

EP is not necessarily another GPU-count multiplier. vLLM documents its EP group as `TP × DP`; verify how PP stages and the installed configuration interact. SGLang distinguishes ordinary DP replicas from `--attn-dp-size` groups inside TP, with constraints on combining those modes. Validate world size, rank placement, and actual groups from logs/source; similarly named flags can have different meanings. Sources: [vLLM EP groups](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments), [SGLang DP/DPA constraints](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide).

## Measure online behavior before selecting a layout

First confirm a real generation and the required API behavior. Then replay the same representative traffic through the serving endpoint and actual gateway/client when used: one active stream, peak concurrency, realistic arrivals/bursts, mixed short/long prompts, full native context, long outputs/reasoning, and cold/warm prefixes. Include tools, structured output, and modalities when required. Control output lengths/sampling and record actual generated tokens. Use the engine benchmark client when it can reproduce this traffic; canned/offline scores remain screening data.

For each candidate, record layout and placement; per-user and aggregate output tokens/s; prefill throughput; successful requests/s; SLO-compliant goodput; TTFT/TPOT and end-to-end p50/p95/p99 with enough samples; per-GPU peak memory; queue/preemption/OOM/error rates; cache hit rate; and communication/stage/expert imbalance. Normalize to the same resource budget and report startup/compile costs separately. Retest model quality where numerical execution changed. Keep profilers off during performance measurements.

Preserve the best runnable configuration after each family. Retry a failure through [issue-first troubleshooting](startup-troubleshooting.md); record unsupported or failed cases rather than hiding them. Stop when the planned matrix or agreed budget is exhausted, or further improvements are below measurement noise. Repeat the promising finalists sufficiently to distinguish noise, then validate the winner with production-like traffic and the intended speculation/cache options before deployment. Report tradeoffs when the single-user latency winner differs from the aggregate-throughput winner.
