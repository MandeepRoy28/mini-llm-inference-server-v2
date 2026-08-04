# Mini LLM Inference Server

> Build a complete LLM inference stack from scratch: KV caching, paged attention, continuous batching, streaming API, benchmark harness.
>
> Stack: PyTorch + FastAPI · Reference: vLLM paper, nanoGPT

---

## How to use

### 1. Work on a problem

Problems are in `problems/`. Each file has a function stub with a full docstring describing what to implement.

Copy the stub to `solutions/` and fill it in:

```bash
cp problems/001_build_token_vocab.py solutions/001_build_token_vocab.py
# open solutions/001_build_token_vocab.py and implement the function
```

### 2. Test your solution

```bash
python submit.py 001 --test-only
```

### 3. Submit (test + commit + push)

```bash
python submit.py 001
```

This runs the tests, commits your solution, and pushes to GitHub — just like DeepML's Submit button.

### 4. Check your progress

```bash
python scaffold.py
```

---

## Setup

```bash
pip install torch transformers fastapi uvicorn pydantic pytest numpy matplotlib
```

Initialize git remote (first time):

```bash
git init
git remote add origin https://github.com/<your-username>/mimi-llm-inference-server.git
git add .
git commit -m "init: project scaffold"
git push -u origin main
```

---

## Parts

| Part | Topic | Problems |
|------|-------|----------|
| 1 | Tiny Transformer (Decoder-Only) | 001–009 |
| 2 | Sampling and Basic Generation | 010–016 |
| 3 | KV Cache | 017–024 |
| 4 | Paged Attention | 025–033 |
| 5 | Continuous Batching | 034–042 |
| 6 | Streaming Serving API | 043–050 |
| 7 | Throughput and Latency Benchmark Harness | 051–058 |

---

## Key Papers

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) — read before Part 4
- [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732) — background on KV cache
- [Continuous Batching — Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference) — read before Part 5
- [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) — reference for Part 1

---

## Progress
**3/58 problems solved**

| # | Problem | Status |
|---|---------|--------|
| | **Part 1 — Tiny Transformer (Decoder-Only)** | |
| 001 | `build_token_vocab` | ✅ |
| 002 | `encode_and_decode` | ✅ |
| 003 | `build_causal_mask` | ✅ |
| 004 | `scaled_dot_product_attention` | ⬜ |
| 005 | `multi_head_attention_forward` | ⬜ |
| 006 | `feed_forward_block` | ⬜ |
| 007 | `transformer_block` | ⬜ |
| 008 | `gpt_model_forward` | ⬜ |
| 009 | `load_pretrained_weights` | ⬜ |
| | **Part 2 — Sampling and Basic Generation** | |
| 010 | `greedy_sample` | ⬜ |
| 011 | `temperature_scaling` | ⬜ |
| 012 | `top_k_filter` | ⬜ |
| 013 | `top_p_nucleus_filter` | ⬜ |
| 014 | `sample_next_token` | ⬜ |
| 015 | `autoregressive_generate` | ⬜ |
| 016 | `generate_with_prompt` | ⬜ |
| | **Part 3 — KV Cache** | |
| 017 | `understand_kv_recompute_cost` | ⬜ |
| 018 | `allocate_kv_cache_buffers` | ⬜ |
| 019 | `write_kv_to_cache` | ⬜ |
| 020 | `read_kv_from_cache` | ⬜ |
| 021 | `attention_with_kv_cache` | ⬜ |
| 022 | `prefill_phase` | ⬜ |
| 023 | `decode_phase` | ⬜ |
| 024 | `benchmark_kv_cache_speedup` | ⬜ |
| | **Part 4 — Paged Attention** | |
| 025 | `understand_memory_fragmentation` | ⬜ |
| 026 | `define_page_and_block_table` | ⬜ |
| 027 | `allocate_page_pool` | ⬜ |
| 028 | `assign_page_to_sequence` | ⬜ |
| 029 | `write_kv_to_page` | ⬜ |
| 030 | `read_kv_via_block_table` | ⬜ |
| 031 | `free_pages_on_completion` | ⬜ |
| 032 | `attention_with_paged_kv` | ⬜ |
| 033 | `benchmark_memory_utilization` | ⬜ |
| | **Part 5 — Continuous Batching** | |
| 034 | `understand_static_batching_problem` | ⬜ |
| 035 | `build_request_queue` | ⬜ |
| 036 | `build_running_batch` | ⬜ |
| 037 | `iteration_level_scheduler` | ⬜ |
| 038 | `batched_prefill` | ⬜ |
| 039 | `batched_decode_step` | ⬜ |
| 040 | `handle_sequence_completion` | ⬜ |
| 041 | `run_continuous_batching_loop` | ⬜ |
| 042 | `benchmark_throughput` | ⬜ |
| | **Part 6 — Streaming Serving API** | |
| 043 | `define_generate_request_schema` | ⬜ |
| 044 | `define_generate_response_schema` | ⬜ |
| 045 | `build_inference_engine_singleton` | ⬜ |
| 046 | `implement_streaming_generator` | ⬜ |
| 047 | `build_sse_streaming_endpoint` | ⬜ |
| 048 | `build_non_streaming_endpoint` | ⬜ |
| 049 | `add_request_id_tracking` | ⬜ |
| 050 | `test_streaming_with_curl` | ⬜ |
| | **Part 7 — Throughput and Latency Benchmark Harness** | |
| 051 | `define_benchmark_metrics` | ⬜ |
| 052 | `build_synthetic_request_generator` | ⬜ |
| 053 | `run_single_request_benchmark` | ⬜ |
| 054 | `run_concurrent_request_benchmark` | ⬜ |
| 055 | `plot_latency_vs_batch_size` | ⬜ |
| 056 | `plot_throughput_vs_concurrency` | ⬜ |
| 057 | `compare_with_without_paged_attention` | ⬜ |
| 058 | `write_benchmark_report` | ⬜ |