# LLGuidance bitmask-fill overhead in real vLLM decode

End-to-end check of a small CPU-side change in how the LLGuidance token bitmask is filled, inside real
vLLM 0.31.0 model decoding on an H100. Outputs are token-identical in every run.

## Variants

All three use vLLM 0.31.0 with `structured_outputs_config={"backend": "guidance"}` and the llguidance
1.7.6 wheel vLLM installs. Only the per-row fill call differs.

- **stock**: unchanged vLLM. `GuidanceGrammar.fill_bitmask` calls `llguidance.torch.fill_next_token_bitmask`,
  which creates a row view `bitmask[index, :]` on every call.
- **lib**: vLLM unchanged; only `llguidance.torch.fill_next_token_bitmask` is replaced by the proposed
  version in `llguidance-pr/pr.diff`. It keeps the same checks but computes the row address
  (`data_ptr() + index * stride(0) * 4`) instead of creating a view.
- **ptr**: vLLM's `StructuredOutputManager._fill_bitmasks` caches the persistent bitmask base pointer and
  row stride once and calls `ll_matcher.unsafe_compute_mask_ptr(base + row * stride, stride)` directly
  (`check_error()` kept).

## Setup

- Model `Qwen/Qwen2.5-0.5B-Instruct`, greedy (temperature 0), max 384 new tokens, prefix caching off,
  `max_num_seqs=64`, in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, so one engine runs all
  variants), async scheduling on (vLLM default).
- Workload: 54 JSONSchemaBench schemas (55 preselected shared-correct schemas from
  `guidance-ai/jsonschemabench@9a94995`; one dropped because its prompt exceeds 6,000 tokens). Prompt asks
  for one JSON value satisfying the schema. Batches 1/4/16/64 as 8/4/2/1 groups of concurrent requests.
- One warmup pass per variant, then 7 repeats, counterbalanced order per group.
- Timed: `llm.generate` wall time (E2E), and wall time inside `StructuredOutputManager.grammar_bitmask`
  per decode step.
- Hardware: Daytona H100 80GB HBM3 sandbox (4 vCPU cap), Intel Xeon Platinum 8468 (run 1) / 8480+ (run 2);
  vLLM 0.31.0, torch 2.13.0+cu130, llguidance 1.7.6, Python 3.12.3 (image `vllm/vllm-openai:v0.31.0`).

## Correctness

Every generated token sequence is identical across all variants and repeats: run 1 240/240 runs, run 2
345/345 runs, and the reference digests of runs 1 and 2 are identical (`outputs_token_identical_all_runs`
and `reference_digests` in each `measurements.json`; texts in `outputs.json`).

## Results (median of 7 paired repeats; [min, max] of paired reductions)

Run 2 (`results/run2-stock-lib-ptr`):

| batch | grammar_bitmask us/step stock / lib / ptr | lib: mask-fill reduction | ptr: mask-fill reduction | mask fill share of E2E (stock) | lib: E2E reduction | ptr: E2E reduction |
|---|---|---|---|---|---|---|
| 1  | 65.9 / 62.0 / 58.9    | 5.51% [4.43, 6.02] | 10.52% [9.65, 11.26] | 4.5%  | 0.33% [-0.83, 0.96] | -0.21% [-0.84, 0.83] |
| 4  | 92.5 / 87.5 / 82.9    | 5.75% [1.95, 7.41] | 11.20% [9.53, 11.96] | 6.0%  | 0.18% [-4.76, 1.48] | 1.14% [0.19, 1.60] |
| 16 | 184.8 / 176.1 / 167.0 | 5.18% [4.11, 7.60] | 9.88% [8.31, 10.44]  | 10.4% | 0.90% [-0.61, 2.25] | 0.89% [0.13, 2.46] |
| 64 | 769.1 / 734.3 / 700.0 | 4.72% [3.16, 6.42] | 8.50% [8.04, 10.97]  | 24.8% | 1.28% [-0.28, 2.87] | 2.33% [1.92, 3.43] |

Run 1 (`results/run1-stock-vs-ptr`, separate sandbox, stock vs ptr only): mask-fill reduction 12.49 / 11.87 /
10.33 / 9.11% and E2E reduction 0.58 / 0.73 / 1.13 / 1.89% at batch 1/4/16/64.

Takeaways:

- At batch 64 with a 0.5B model, LLGuidance mask filling is ~25% of decode step wall time in vLLM.
- Removing per-call Python/tensor overhead cuts that fill time by 8.5–12.5% (ptr) or ~5% (library-only fix).
- End-to-end the gain is small: ~1–2.3% at batch 16–64, within noise at batch 1–4.

## llguidance helper change (`llguidance-pr/`)

`pr.diff` against `guidance-ai/llguidance@e7c6900`: `fill_next_token_bitmask` computes the row pointer
instead of slicing; same dtype/CPU/2-D/contiguity checks; out-of-range rows still raise `IndexError`
(negative indices still wrap). Adds `test_fill_bitmask_row_addressing` (plain/negative/offset/row-strided
views, bounds, non-contiguous rejection). On the released 1.9.1 wheel (whose `torch.py` equals upstream
`main`) `test_hf.py` + `test_bitmask.py` pass before (10 passed, 2 skipped) and after the change (10 passed,
2 skipped: mlx and CUDA unavailable). CPU micro-benchmark on a mid-string JSON state with the Qwen2.5
vocabulary: 3.633 -> 2.732 us per call (`micro.txt`).

## Reproduce

```sh
# H100, image vllm/vllm-openai:v0.31.0
pip install jsonschema
python3 bench_vllm_e2e.py --out out-run2 --modes stock,lib,ptr --repeats 7 --max-tokens 384
python3 bench_vllm_e2e_run1.py --out out-run1 --repeats 7 --max-tokens 384   # run 1 (stock vs ptr)
bash llguidance-pr/run_tests.sh   # CPU; expects torch_orig.py / torch_patched.py / tests alongside
```

Raw logs for both GPU runs and the CPU test run are in `logs/`. Not measured: other models, batch > 128
(vLLM switches to a thread-pool fill path), speculative decoding, the multiprocess engine, SGLang.
