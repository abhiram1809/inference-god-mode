# API verification

Check the installed server's route table and documentation; endpoint names alone do not establish compatibility. Make requests through each real client protocol required by the user:

| Protocol | Minimal route | Additional checks when needed |
| --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | Streaming chunks, tool calls/results, JSON schema, reasoning fields, usage, stop conditions. |
| OpenAI Responses | `POST /v1/responses` | SSE event sequence, function-call continuation, `previous_response_id` or state if client uses it, usage and errors. |
| Anthropic Messages | `POST /v1/messages` | Content blocks, `tool_use`/`tool_result`, SSE, `max_tokens`, `stop_reason`, token counting if client needs it. |

Use the model's tokenizer and chat template. Enable the backend's model-specific tool and reasoning parsers only after confirming the correct values in its current docs. Verify behavior with a real tool round trip, multi-turn exchange, long-context request, and malformed request. A 200 response with plain text is insufficient for an agent harness. If an adapter translates protocols, test semantics and streaming across the adapter and state the extra component in the deployment.

For externally reachable serving, include an authenticated front end, TLS, rate/concurrency limits, and client timeouts. Keep exploratory servers on loopback until the user requests network exposure.

Primary references: [vLLM online serving](https://docs.vllm.ai/en/latest/serving/online_serving/), [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [TensorRT-LLM serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html), [MLX-LM server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md).
