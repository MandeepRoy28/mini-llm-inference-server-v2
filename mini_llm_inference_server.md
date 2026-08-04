# Mini LLM Inference Server — Step-by-Step Roadmap
> Build a complete LLM inference stack from scratch: KV caching, paged attention, continuous batching, streaming API, benchmark harness.
> Stack: PyTorch + FastAPI
> Reference: vLLM paper, nanoGPT

---

## Part 1 — Tiny Transformer (Decoder-Only)
> Build a minimal GPT-style decoder-only transformer to serve as the model under the inference stack.

- [ ] `001` `build_token_vocab` — Build token→id and id→token dicts from a small text corpus
- [ ] `002` `encode_and_decode` — Encode string to token ids; decode ids back to string
- [ ] `003` `build_causal_mask` — Upper-triangular mask blocking future token positions
- [ ] `004` `scaled_dot_product_attention` — Q, K, V matmul with scale + causal mask + softmax
- [ ] `005` `multi_head_attention_forward` — Split into heads, attend, merge, output project
- [ ] `006` `feed_forward_block` — Linear → GELU → Linear with residual
- [ ] `007` `transformer_block` — MHA + FFN with pre-LayerNorm (GPT-2 style)
- [ ] `008` `gpt_model_forward` — Embed tokens + positional encoding → N blocks → unembed to logits
- [ ] `009` `load_pretrained_weights` — Load DistilGPT2 weights from HuggingFace into your model structure

---

## Part 2 — Sampling and Basic Generation
> Implement the core token sampling methods used in inference.

- [ ] `010` `greedy_sample` — Pick argmax of logits at each step
- [ ] `011` `temperature_scaling` — Divide logits by temperature T before softmax
- [ ] `012` `top_k_filter` — Zero out all logits except top-k before sampling
- [ ] `013` `top_p_nucleus_filter` — Zero out logits outside the nucleus (cumulative prob > p)
- [ ] `014` `sample_next_token` — Combine temperature + top_k + top_p into one sampling function
- [ ] `015` `autoregressive_generate` — Loop: forward → sample → append → repeat until EOS or max_len
- [ ] `016` `generate_with_prompt` — Encode prompt → generate → decode and return full text

---

## Part 3 — KV Cache
> Cache key and value tensors across decoding steps to avoid recomputing them each step.

- [ ] `017` `understand_kv_recompute_cost` — Benchmark generation without cache; measure time per token
- [ ] `018` `allocate_kv_cache_buffers` — Pre-allocate K and V tensors of shape (batch, n_heads, max_seq, d_k)
- [ ] `019` `write_kv_to_cache` — At step t, write current K and V into cache at position t
- [ ] `020` `read_kv_from_cache` — Read cached K and V up to step t for attention computation
- [ ] `021` `attention_with_kv_cache` — Only compute Q for current token; use full cached K, V
- [ ] `022` `prefill_phase` — Process full prompt in one forward pass; populate KV cache for all prompt tokens
- [ ] `023` `decode_phase` — Autoregressive loop using KV cache; one token per step
- [ ] `024` `benchmark_kv_cache_speedup` — Compare tokens/sec with and without KV cache

---

## Part 4 — Paged Attention
> Manage KV cache memory in fixed-size pages to avoid fragmentation (vLLM's core innovation).

- [ ] `025` `understand_memory_fragmentation` — Simulate naive KV cache allocation; show wasted memory with variable-length sequences
- [ ] `026` `define_page_and_block_table` — A page = fixed-size chunk of KV cache (e.g. 16 tokens); block_table maps sequence → list of page ids
- [ ] `027` `allocate_page_pool` — Pre-allocate a pool of N pages in GPU memory
- [ ] `028` `assign_page_to_sequence` — On new request, assign free pages from pool; update block_table
- [ ] `029` `write_kv_to_page` — Write K, V of new tokens into their assigned page at correct offset
- [ ] `030` `read_kv_via_block_table` — Gather K, V across potentially non-contiguous pages using block_table
- [ ] `031` `free_pages_on_completion` — Return pages to pool when a sequence finishes generation
- [ ] `032` `attention_with_paged_kv` — Run attention using gathered non-contiguous K, V pages
- [ ] `033` `benchmark_memory_utilization` — Compare memory waste: naive allocation vs paged; show GPU utilization improvement

---

## Part 5 — Continuous Batching
> Serve multiple requests simultaneously; add new requests mid-batch without waiting for others to finish.

- [ ] `034` `understand_static_batching_problem` — Show how static batching wastes GPU when sequences finish at different times
- [ ] `035` `build_request_queue` — A queue holding incoming generation requests with their prompt + params
- [ ] `036` `build_running_batch` — A set of in-flight sequences currently being decoded
- [ ] `037` `iteration_level_scheduler` — Each decoding step: check queue, add new requests to batch if capacity allows
- [ ] `038` `batched_prefill` — Prefill multiple new prompts together in one forward pass
- [ ] `039` `batched_decode_step` — One decode step across all in-flight sequences simultaneously
- [ ] `040` `handle_sequence_completion` — Detect EOS; remove finished sequences; free their pages; pull next from queue
- [ ] `041` `run_continuous_batching_loop` — Full scheduler loop: prefill new → decode all → evict finished → repeat
- [ ] `042` `benchmark_throughput` — Measure requests/sec and GPU utilization vs static batching baseline

---

## Part 6 — Streaming Serving API
> Wrap the inference engine in a FastAPI server that streams tokens as they are generated.

- [ ] `043` `define_generate_request_schema` — Pydantic model: prompt, max_tokens, temperature, top_k, top_p
- [ ] `044` `define_generate_response_schema` — Pydantic model: generated_text, tokens_generated, time_to_first_token, total_time
- [ ] `045` `build_inference_engine_singleton` — Load model + allocate page pool once at server startup
- [ ] `046` `implement_streaming_generator` — Python async generator that yields one token at a time as decoded
- [ ] `047` `build_sse_streaming_endpoint` — FastAPI route returning StreamingResponse with text/event-stream content type
- [ ] `048` `build_non_streaming_endpoint` — FastAPI route returning full completed text in one response
- [ ] `049` `add_request_id_tracking` — Assign unique id to each request; log time-to-first-token and total latency
- [ ] `050` `test_streaming_with_curl` — Verify SSE streaming works end-to-end with a curl command

---

## Part 7 — Throughput and Latency Benchmark Harness
> Measure and report key inference performance metrics under load.

- [ ] `051` `define_benchmark_metrics` — Time to first token (TTFT), time per output token (TPOT), throughput (tokens/sec), memory used
- [ ] `052` `build_synthetic_request_generator` — Generate N requests with random prompt lengths and max_token budgets
- [ ] `053` `run_single_request_benchmark` — Measure TTFT and TPOT for one request at a time
- [ ] `054` `run_concurrent_request_benchmark` — Fire N requests simultaneously using asyncio; measure total throughput
- [ ] `055` `plot_latency_vs_batch_size` — Sweep batch sizes 1, 2, 4, 8, 16; plot TTFT and TPOT
- [ ] `056` `plot_throughput_vs_concurrency` — Sweep concurrent users; find saturation point where throughput plateaus
- [ ] `057` `compare_with_without_paged_attention` — Side-by-side memory and throughput comparison
- [ ] `058` `write_benchmark_report` — Print a summary table: config, throughput, p50/p95/p99 latency, GPU memory used

---

## Checkpoints — Test Yourself After Each Part

| After Part | What to verify |
|------------|----------------|
| 1 | Model loads DistilGPT2 weights; generates coherent text |
| 2 | Temperature=0 → same output always; top_p=0.9 → diverse outputs |
| 3 | KV cache gives identical outputs as no-cache but 3-5x faster |
| 4 | Paged attention gives same outputs; memory waste drops significantly |
| 5 | Continuous batching serves 10 concurrent requests without stalling |
| 6 | curl to /generate streams tokens visibly one by one |
| 7 | Benchmark report shows throughput curve and latency percentiles |

---

## Key Papers to Read First
- [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) — read before Part 4
- [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732) — background on KV cache
- [Continuous Batching blog — Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference) — read before Part 5
- nanoGPT by Karpathy — reference implementation for Part 1
