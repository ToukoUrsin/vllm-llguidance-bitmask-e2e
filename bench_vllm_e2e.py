#!/usr/bin/env python3
"""Session 017: real vLLM model decode, stock LLGuidance bitmask fill vs fixed-pointer fill.

One in-process vLLM engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) so both variants share the
same model, CUDA graphs and scheduler. The only difference between variants is
StructuredOutputManager._fill_bitmasks:
  stock: vLLM's own loop -> GuidanceGrammar.fill_bitmask -> llguidance.torch.fill_next_token_bitmask
  ptr:   same loop, but the bitmask base pointer/stride is cached once and each row calls
         ll_matcher.unsafe_compute_mask_ptr(base + row*stride, stride) directly (check_error kept).
Greedy decoding, prefix caching off; outputs must be token-identical across variants and repeats.
Timed: E2E llm.generate wall time per batch group, and per-step grammar_bitmask time.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import llguidance  # noqa: E402
import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
import vllm.v1.structured_output as so  # noqa: E402
from vllm.v1.structured_output.backend_guidance import GuidanceGrammar  # noqa: E402

try:
    from vllm.sampling_params import StructuredOutputsParams as SOP
    SO_KW = "structured_outputs"
except ImportError:  # older vLLM
    from vllm.sampling_params import GuidedDecodingParams as SOP
    SO_KW = "guided_decoding"

MGR = so.StructuredOutputManager
ORIG_FILL = MGR._fill_bitmasks
ORIG_GB = MGR.grammar_bitmask
STEPS = []  # (ns, rows) per grammar_bitmask call
COUNTS = {"ptr_rows": 0}


def fill_ptr(self, batch):
    bm = self._grammar_bitmask
    cache = getattr(self, "_s017_ptr", None)
    if cache is None or cache[0] is not bm:
        assert bm.dtype == torch.int32 and bm.is_cpu and bm.dim() == 2 and bm.is_contiguous()
        cache = self._s017_ptr = (bm, bm.data_ptr(), bm.shape[1] * 4)
    _, base, stride = cache
    for grammar, index, apply_bitmask in batch:
        if apply_bitmask and not grammar.is_terminated():
            if type(grammar) is GuidanceGrammar:
                grammar.ll_matcher.unsafe_compute_mask_ptr(base + index * stride, stride)
                grammar.check_error()
                COUNTS["ptr_rows"] += 1
            else:
                grammar.fill_bitmask(bm, index)
        else:
            bm[index].fill_(self._full_mask)


def timed_gb(self, requests, structured_output_request_ids, *a, **k):
    t = time.perf_counter_ns()
    r = ORIG_GB(self, requests, structured_output_request_ids, *a, **k)
    STEPS.append((time.perf_counter_ns() - t, len(structured_output_request_ids)))
    return r


import llguidance.torch as llt  # noqa: E402
ORIG_HELPER = llt.fill_next_token_bitmask


def helper_ptr(interp, bitmask, index=0):
    """Proposed llguidance.torch.fill_next_token_bitmask: same checks, no per-call row view."""
    assert bitmask.dtype == torch.int32, "Mask must be int32"
    assert bitmask.is_cpu, "Mask must be on CPU"
    assert bitmask.dim() == 2, "Mask must be 2D"
    rows, words = bitmask.shape
    if index < 0:
        index += rows
    assert 0 <= index < rows, "Index out of range"
    assert words == 1 or bitmask.stride(1) == 1, "Mask must be contiguous"
    COUNTS["ptr_rows"] += 1
    interp.unsafe_compute_mask_ptr(bitmask.data_ptr() + index * bitmask.stride(0) * 4, words * 4)


MGR.grammar_bitmask = timed_gb
# stock: vLLM + released llguidance unchanged; lib: only llguidance.torch helper replaced;
# ptr: vLLM-side loop with cached base pointer (helper untouched).
MODES = {"stock": (ORIG_FILL, ORIG_HELPER), "lib": (ORIG_FILL, helper_ptr), "ptr": (fill_ptr, ORIG_HELPER)}
ACTIVE = ["stock", "ptr"]


import vllm.v1.structured_output.backend_guidance as bg  # noqa: E402
_ = bg.llguidance_torch.allocate_token_bitmask  # force vLLM's LazyLoader to load (it copies attributes)


def set_mode(m):
    MGR._fill_bitmasks, helper = MODES[m]
    llt.fill_next_token_bitmask = helper
    setattr(bg.llguidance_torch, "fill_next_token_bitmask", helper)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--schemas", default="schemas.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--groups", default="8,4,2,1", help="groups per batch size")
    ap.add_argument("--limit-schemas", type=int, default=0)
    ap.add_argument("--modes", default="stock,ptr")
    ap.add_argument("--async-scheduling", default="default", choices=["default", "on", "off"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ACTIVE[:] = args.modes.split(",")
    assert ACTIVE[0] == "stock"

    bundle = json.load(open(args.schemas))
    schemas = bundle["schemas"]
    if args.limit_schemas:
        schemas = schemas[: args.limit_schemas]
    kw = dict(model=args.model, seed=0, max_model_len=8192, max_num_seqs=64, gpu_memory_utilization=0.6,
              enable_prefix_caching=False, structured_outputs_config={"backend": "guidance"})
    if args.async_scheduling != "default":
        kw["async_scheduling"] = args.async_scheduling == "on"
    try:
        llm = LLM(**kw)
    except TypeError:
        kw.pop("structured_outputs_config")
        kw["guided_decoding_backend"] = "guidance"
        llm = LLM(**kw)
    tok = llm.get_tokenizer()
    cfg = llm.llm_engine.vllm_config if hasattr(llm.llm_engine, "vllm_config") else None
    async_sched = getattr(getattr(cfg, "scheduler_config", None), "async_scheduling", None)

    def prompt(schema):
        msgs = [{"role": "system", "content": "You output only JSON."},
                {"role": "user", "content": "Return one JSON value that satisfies this JSON Schema. Output only the JSON.\n\n"
                 + json.dumps(schema)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def sp(schema, max_tokens):
        return SamplingParams(temperature=0.0, max_tokens=max_tokens, **{SO_KW: SOP(json=schema)})

    # Screen: keep schemas vLLM's guidance backend accepts and whose prompt fits.
    set_mode("stock")
    usable, rejected = [], []
    for s in schemas:
        p = prompt(s["schema"])
        if len(tok.encode(p)) > 6000:
            rejected.append({"case": s["case"], "why": "prompt>6000 tokens"})
            continue
        try:
            llm.generate([p], [sp(s["schema"], 4)], use_tqdm=False)
            usable.append(dict(s, prompt=p))
        except Exception as e:
            rejected.append({"case": s["case"], "why": f"{type(e).__name__}: {str(e)[:200]}"})
    print(f"usable {len(usable)} rejected {len(rejected)}", flush=True)
    order = sorted(usable, key=lambda s: hashlib.sha256(("s017:" + s["case"]).encode()).hexdigest())
    n = len(order)

    groups = []
    for b, g in zip([int(x) for x in args.batches.split(",")], [int(x) for x in args.groups.split(",")]):
        for gi in range(g):
            idx = [(gi * b + j) % n for j in range(b)]
            groups.append({"batch": b, "group": gi, "idx": idx})

    def run(group, mode):
        set_mode(mode)
        items = [order[i] for i in group["idx"]]
        STEPS.clear()
        COUNTS["ptr_rows"] = 0
        torch.cuda.synchronize()
        t = time.perf_counter_ns()
        outs = llm.generate([s["prompt"] for s in items], [sp(s["schema"], args.max_tokens) for s in items], use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter_ns() - t
        toks = [list(o.outputs[0].token_ids) for o in outs]
        if mode != "stock":
            assert COUNTS["ptr_rows"] > 0, "patched path not exercised"
        return {"wall_ns": wall, "gb_ns": sum(s[0] for s in STEPS), "gb_calls": len(STEPS),
                "gb_rows": sum(s[1] for s in STEPS), "out_tokens": sum(len(x) for x in toks),
                "digest": hashlib.sha256(json.dumps(toks).encode()).hexdigest(), "toks": toks,
                "texts": [o.outputs[0].text for o in outs], "finish": [o.outputs[0].finish_reason for o in outs]}

    # Warmup both modes on every group (also first equality check).
    ref = {}
    for g in groups:
        for m in ACTIVE:
            r = run(g, m)
            key = (g["batch"], g["group"])
            ref.setdefault(key, r)
            assert r["digest"] == ref[key]["digest"], f"warmup mismatch {key} {m}"
    print("warmup ok", flush=True)

    rows = []
    mismatches = 0
    for rep in range(args.repeats):
        for gi, g in enumerate(groups):
            k = (rep + gi) % len(ACTIVE)
            modes = ACTIVE[k:] + ACTIVE[:k]
            if rep % 2:
                modes = modes[::-1]
            for m in modes:
                r = run(g, m)
                key = (g["batch"], g["group"])
                same = r["digest"] == ref[key]["digest"]
                mismatches += 0 if same else 1
                rows.append({"rep": rep, "batch": g["batch"], "group": g["group"], "mode": m, "identical": same,
                             **{k: r[k] for k in ("wall_ns", "gb_ns", "gb_calls", "gb_rows", "out_tokens", "digest")}})
        print(f"rep {rep} done", flush=True)

    # JSON validity of the (identical) outputs, informational.
    valid = {}
    try:
        import jsonschema
        for key, r in ref.items():
            ok = 0
            g = next(x for x in groups if (x["batch"], x["group"]) == key)
            for text, i in zip(r["texts"], g["idx"]):
                try:
                    jsonschema.validate(json.loads(text), order[i]["schema"])
                    ok += 1
                except Exception:
                    pass
            valid[f"{key[0]}-{key[1]}"] = [ok, len(g["idx"])]
    except ImportError:
        valid = None

    summary = []
    for b in sorted({g["batch"] for g in groups}):
        per = {}
        for m in ACTIVE:
            sel = [r for r in rows if r["batch"] == b and r["mode"] == m]
            reps = sorted({r["rep"] for r in sel})
            per[m] = {"wall": [sum(r["wall_ns"] for r in sel if r["rep"] == k) for k in reps],
                      "gb": [sum(r["gb_ns"] for r in sel if r["rep"] == k) for k in reps],
                      "calls": sum(r["gb_calls"] for r in sel if r["rep"] == reps[0]),
                      "rows": sum(r["gb_rows"] for r in sel if r["rep"] == reps[0])}
        s0 = per["stock"]
        row = {"batch": b, "groups": len({r["group"] for r in rows if r["batch"] == b}),
               "decode_steps_per_repeat": s0["calls"], "mask_rows_per_repeat": s0["rows"],
               "out_tokens_per_repeat": sum(r["out_tokens"] for r in rows if r["batch"] == b and r["rep"] == 0 and r["mode"] == "stock"),
               "e2e_ms_median": {m: statistics.median(per[m]["wall"]) / 1e6 for m in ACTIVE},
               "e2e_us_per_step_median": {m: statistics.median(per[m]["wall"]) / 1e3 / per[m]["calls"] for m in ACTIVE},
               "grammar_bitmask_us_per_step_median": {m: statistics.median(per[m]["gb"]) / 1e3 / per[m]["calls"] for m in ACTIVE},
               "grammar_bitmask_us_per_row_median": {m: statistics.median(per[m]["gb"]) / 1e3 / per[m]["rows"] for m in ACTIVE},
               "grammar_share_of_e2e_pct_stock": 100 * statistics.median(s0["gb"]) / statistics.median(s0["wall"])}
        for m in ACTIVE[1:]:
            pw = [100 * (a - c) / a for a, c in zip(s0["wall"], per[m]["wall"])]
            pg = [100 * (a - c) / a for a, c in zip(s0["gb"], per[m]["gb"])]
            row[f"{m}_e2e_reduction_pct_median_paired"] = statistics.median(pw)
            row[f"{m}_e2e_reduction_pct_paired_min_max"] = [min(pw), max(pw)]
            row[f"{m}_grammar_bitmask_reduction_pct_median_paired"] = statistics.median(pg)
            row[f"{m}_grammar_bitmask_reduction_pct_paired_min_max"] = [min(pg), max(pg)]
        summary.append(row)

    out = {
        "session": "017-vllm-e2e",
        "modes": ACTIVE,
        "model": args.model, "max_tokens": args.max_tokens, "repeats": args.repeats,
        "schemas_source": bundle["source"], "schemas_selection": bundle["selection"],
        "usable_schemas": len(usable), "rejected": rejected,
        "outputs_token_identical_all_runs": mismatches == 0, "mismatched_runs": mismatches,
        "runs": len(rows) + 2 * len(groups),
        "json_schema_valid_outputs": valid,
        "summary": summary, "rows": rows,
        "reference_digests": {f"{k[0]}-{k[1]}": v["digest"] for k, v in ref.items()},
        "config": {"VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
                   "async_scheduling": async_sched, "prefix_caching": False, "max_num_seqs": 64,
                   "temperature": 0.0, "backend": "guidance", "structured_kw": SO_KW},
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__,
                     "llguidance": getattr(llguidance, "__version__", sh("pip show llguidance | grep Version")),
                     "python": platform.python_version()},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cpu": sh("grep -m1 'model name' /proc/cpuinfo"),
                     "nproc": os.cpu_count(), "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")},
    }
    json.dump(out, open(os.path.join(args.out, "measurements.json"), "w"), indent=1)
    json.dump({k: v["texts"] for k, v in {f"{k[0]}-{k[1]}": v for k, v in ref.items()}.items()},
              open(os.path.join(args.out, "outputs.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("outputs_token_identical_all_runs", "mismatched_runs", "usable_schemas", "summary", "config", "versions", "hardware")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
