# Organizational and Kubernetes serving

Read for organizational multi-machine deployments. Keep the optimized engine configuration as a baseline, then tune routing and fleet architecture against real traffic. **llm-d** is an orchestration layer above an engine; retain the allowed engine choices and verify the selected integration's actual features.

## Decide whether the additional architecture pays

Capture weekly active users, peak active generations, arrival/burst rates, input/output token distributions, repeated-prefix patterns, TTFT/TPOT SLOs, availability target, GPU budget, and operating capacity. Inspect the existing Kubernetes environment, GPU resources, node/network topology, and measured cross-node bandwidth/latency.

Use the user's **approximately 500–600 weekly active users** guideline as a default complexity heuristic: below that range, usually keep a tuned colocated engine or modest replica deployment and defer distributed prefill/decode. This is a user preference, not an upstream threshold or capacity formula. A few long-context agents can generate heavy traffic; large user counts can generate little traffic. Any exception should show measured interference, required availability/capacity, and the cost benefit. Existing Kubernetes use alone does not justify disaggregation.

Distinguish three decisions: replicating complete model servers for throughput/availability; sharding one model across GPUs/nodes for fit or performance; and separating prefill from decode. They can compose, but solve different constraints.

## Introduce llm-d routing before splitting stages

Use a current [llm-d deployment recipe](https://github.com/llm-d/llm-d) and inspect its versioned [architecture](https://llm-d.ai/docs/architecture): gateway/proxy, Endpoint Picker (EPP), InferencePool, and model-server pods. Pin charts/manifests, CRD versions, router/sidecar images, engine images, and checkpoint revisions. Do not invent manifests from a stale API example.

Compare ordinary round-robin/load routing with **prefix-cache affinity plus load awareness**. Repeated system prompts, documents, and conversation history can benefit from routing to a pod holding the reusable prefix. Include queue/token load and KV pressure so affinity does not overload a hot pod. The [optimized baseline](https://llm-d.ai/docs/well-lit-paths/foundations/optimized-baseline) supports cache- and load-aware selection; use the installed version's scorer configuration.

For [precise cache routing](https://llm-d.ai/docs/well-lit-paths/foundations/precise-prefix-cache-routing), verify engine KV-event support, subscriptions, compatible tokenization/block keys, and index freshness. Measure approximate versus precise routing when useful. Routing to a cache owner does not itself transfer KV between pods. Preserve tenant cache isolation and test the required OpenAI/Anthropic dialects, streaming, tools, and reasoning through the gateway; engine endpoint support does not establish router support.

## Disaggregate prefill and decode when interference binds

Long prefills consume substantial compute and can interrupt active decodes, worsening inter-token latency under mixed traffic. Separate pools can isolate those workloads and let each stage use an appropriate batch/parallelism policy. Validate this mechanism with the [llm-d P/D path](https://llm-d.ai/docs/well-lit-paths/foundations/pd-disaggregation), rather than inferring fleet performance from an isolated vLLM benchmark.

Request flow: gateway/EPP selects endpoints → prefill worker builds KV → supported connector transfers KV → decode worker generates and streams. Check exact model/checkpoint, cache layout/dtype/scales, connector, engine revisions, and supported TP layouts on both sides. Recheck FP8 KV, hybrid attention, multimodal state, and speculation compatibility. Account for model copies, full-context KV, and workspace in each pool.

For prompt-heavy traffic, test **more prefill workers with one or two decode workers** as an initial hypothesis. Worker, GPU, and machine counts differ when TP spans devices. Size prefill from uncached input-token demand and decode from output-token demand, active streams, and KV capacity; long reasoning/output traffic may need more decode capacity. Sweep the pool ratio while meeting both latency targets.

Measure KV bytes, transfer time, network contention, and waiting time between stages. High-bandwidth networking such as InfiniBand, RoCE, or EFA is recommended by the P/D guide; a supported TCP path still needs workload validation. Compare transfer cost with the prefill/decode interference removed. Separation can lose on short prompts, small load, slow links, or poorly balanced pools.

## Operate and qualify the fleet

- Place GPU workers according to their required devices and topology; identify TP/EP gangs separately from replica counts. Ensure model storage, startup probes, and readiness allow weight loading and kernel warmup. Drain streaming requests during rollouts; test worker/router loss and bounded recovery at the required availability level.
- Scale prefill and decode independently using relevant stage queues, input-token demand, active decodes, and KV pressure. Inspect the current [llm-d autoscaling path](https://llm-d.ai/docs/well-lit-paths/foundations/workload-autoscaling) before choosing HPA/KEDA or another supported controller. Include model startup time, cache warmup, minimum warm capacity, and backpressure; GPU utilization alone is insufficient.
- Compare three candidates at the same GPU budget: colocated replicas with ordinary routing, replicas with cache/load-aware routing, and routed P/D pools. Replay representative traffic, including bursts, mixed long/short prompts, cold/warm prefixes, full-context requests, and peak concurrency. Record gateway TTFT/TPOT p50/p95/p99, SLO-compliant goodput, errors, per-stage queues, cache hits, KV transfer time, memory, and cost per successful request or token.
- Produce versioned manifests/Helm values, topology and pool sizing, tested client endpoints, monitoring signals, rollout/rollback procedure, and benchmark evidence. Render and validate the deployment against its target API versions. Retain the simplest architecture that meets the organization's requirements with a measured benefit.
