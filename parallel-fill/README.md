# vLLM guidance backend: parallel bitmask fill (LLExecutor)

Evidence for the vLLM PR that fills all guidance bitmask rows of a non-spec batch with one
`llguidance.torch.fill_next_token_bitmask_par` call on a shared `LLExecutor`.

Setup: `vllm/vllm-openai:cu129-nightly-5555a2d8f7116b2b4877f66516f4128176cf722a` (structured_output files equal
vLLM main 6849ca69), llguidance 1.7.6, Daytona H100 80GB with 16 vCPUs, Qwen2.5-0.5B-Instruct
(rev 7ae55760), 54 JSONSchemaBench schemas (same set as `../schemas.json`), greedy, max 384 tokens,
prefix caching off, in-process engine, 7 paired repeats with rotated mode order.

Modes: `stock` = upstream `grammar_bitmask` (serial per-row fill), `par` = PR with default `LLExecutor()`,
`par4` = PR with `LLExecutor(num_threads=4)`. Outputs were token-identical in all 360 runs.

Run: `bash run.sh OUT --repeats 7 --max-tokens 384` inside the image. `measurements.json` has every row.
