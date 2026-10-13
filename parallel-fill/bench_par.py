#!/usr/bin/env python3
"""Session 032: real vLLM decode, stock serial LLGuidance bitmask fill vs the PR's parallel fill.

Runs inside the vLLM nightly image whose vllm/v1/structured_output/{__init__,backend_guidance}.py
equal vLLM main 6849ca69 (checked by sha256 before patching). The PR files are copied over the
installed package; the stock grammar_bitmask is loaded from the unmodified upstream __init__.py,
so both modes share one in-process engine (same model, CUDA graphs, scheduler) and alternate per group.
  stock: upstream StructuredOutputManager.grammar_bitmask (serial per-row fill below 128 rows)
  par:   PR grammar_bitmask -> GuidanceBackend.fill_bitmasks -> fill_next_token_bitmask_par,
         LLExecutor() default threads
  par4:  same, LLExecutor(num_threads=4)
Greedy decoding, prefix caching off; outputs must be token-identical across modes and repeats.
"""
import argparse
import hashlib
import importlib.util
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
from vllm.sampling_params import StructuredOutputsParams as SOP  # noqa: E402
import vllm.v1.structured_output as so  # noqa: E402
import vllm.v1.structured_output.backend_guidance as bg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("so_orig", os.path.join(HERE, "vllm-orig", "__init__.py"))
so_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(so_orig)

MGR = so.StructuredOutputManager
assert hasattr(bg.GuidanceBackend, "fill_bitmasks"), "PR backend_guidance.py not installed"
GB = {"stock": so_orig.StructuredOutputManager.grammar_bitmask, "par": MGR.grammar_bitmask}
STEPS = []  # (ns, rows) per grammar_bitmask call
COUNTS = {"par_calls": 0}
STATE = {"gb": GB["stock"], "executor": None}
EXECUTORS = {}
ORIG_FILL = bg.GuidanceBackend.fill_bitmasks


def counted_fill(self, bitmask, batch):
    COUNTS["par_calls"] += 1
    self._ll_executor = STATE["executor"]
    return ORIG_FILL(self, bitmask, batch)


def timed_gb(self, requests, ids, *a, **k):
    t = time.perf_counter_ns()
    r = STATE["gb"](self, requests, ids, *a, **k)
    STEPS.append((time.perf_counter_ns() - t, len(ids)))
    return r


MGR.grammar_bitmask = timed_gb
bg.GuidanceBackend.fill_bitmasks = counted_fill
ACTIVE = ["stock", "par", "par4"]


def set_mode(m):
    STATE["gb"] = GB["stock" if m == "stock" else "par"]
    if m != "stock":
        if m not in EXECUTORS:
            EXECUTORS[m] = llguidance.LLExecutor(num_threads=int(m[3:])) if m != "par" else llguidance.LLExecutor()
        STATE["executor"] = EXECUTORS[m]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    ap.add_argument("--schemas", default="schemas.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--groups", default="8,4,2,1", help="groups per batch size")
    ap.add_argument("--limit-schemas", type=int, default=0)
    ap.add_argument("--modes", default="stock,par,par4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ACTIVE[:] = args.modes.split(",")
    assert ACTIVE[0] == "stock"

    bundle = json.load(open(args.schemas))
    schemas = bundle["schemas"]
    if args.limit_schemas:
        schemas = schemas[: args.limit_schemas]
    llm = LLM(model=args.model, revision=args.revision, seed=0, max_model_len=8192, max_num_seqs=64,
              gpu_memory_utilization=0.6, enable_prefix_caching=False,
              structured_outputs_config={"backend": "guidance"})
    tok = llm.get_tokenizer()
    cfg = llm.llm_engine.vllm_config
    async_sched = getattr(cfg.scheduler_config, "async_scheduling", None)

    def prompt(schema):
        msgs = [{"role": "system", "content": "You output only JSON."},
                {"role": "user", "content": "Return one JSON value that satisfies this JSON Schema. Output only the JSON.\n\n"
                 + json.dumps(schema)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def sp(schema, max_tokens):
        return SamplingParams(temperature=0.0, max_tokens=max_tokens, structured_outputs=SOP(json=schema))

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
            groups.append({"batch": b, "group": gi, "idx": [(gi * b + j) % n for j in range(b)]})

    def run(group, mode):
        set_mode(mode)
        items = [order[i] for i in group["idx"]]
        STEPS.clear()
        COUNTS["par_calls"] = 0
        torch.cuda.synchronize()
        t = time.perf_counter_ns()
        outs = llm.generate([s["prompt"] for s in items], [sp(s["schema"], args.max_tokens) for s in items], use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter_ns() - t
        toks = [list(o.outputs[0].token_ids) for o in outs]
        if mode == "stock":
            assert COUNTS["par_calls"] == 0, "stock mode used the parallel path"
        else:
            assert COUNTS["par_calls"] > 0, "parallel path not exercised"
        return {"wall_ns": wall, "gb_ns": sum(s[0] for s in STEPS), "gb_calls": len(STEPS),
                "gb_rows": sum(s[1] for s in STEPS), "par_calls": COUNTS["par_calls"],
                "out_tokens": sum(len(x) for x in toks),
                "digest": hashlib.sha256(json.dumps(toks).encode()).hexdigest(), "toks": toks,
                "texts": [o.outputs[0].text for o in outs], "finish": [o.outputs[0].finish_reason for o in outs]}

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
                             **{k: r[k] for k in ("wall_ns", "gb_ns", "gb_calls", "gb_rows", "par_calls", "out_tokens", "digest")}})
        print(f"rep {rep} done", flush=True)

    valid = {}
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
               "grammar_share_of_e2e_pct": {m: 100 * statistics.median(per[m]["gb"]) / statistics.median(per[m]["wall"]) for m in ACTIVE}}
        for m in ACTIVE[1:]:
            pw = [100 * (a - c) / a for a, c in zip(s0["wall"], per[m]["wall"])]
            pg = [100 * (a - c) / a for a, c in zip(s0["gb"], per[m]["gb"])]
            row[f"{m}_e2e_reduction_pct_median_paired"] = statistics.median(pw)
            row[f"{m}_e2e_reduction_pct_paired_min_max"] = [min(pw), max(pw)]
            row[f"{m}_grammar_bitmask_reduction_pct_median_paired"] = statistics.median(pg)
            row[f"{m}_grammar_bitmask_reduction_pct_paired_min_max"] = [min(pg), max(pg)]
            row[f"{m}_throughput_speedup_x"] = statistics.median(s0["wall"]) / statistics.median(per[m]["wall"])
        summary.append(row)

    out = {
        "session": "032-vllm-par",
        "modes": ACTIVE, "model": args.model, "revision": args.revision,
        "max_tokens": args.max_tokens, "repeats": args.repeats,
        "schemas_source": bundle["source"], "schemas_selection": bundle["selection"],
        "usable_schemas": len(usable), "rejected": rejected,
        "outputs_token_identical_all_runs": mismatches == 0, "mismatched_runs": mismatches,
        "runs": len(rows) + len(ACTIVE) * len(groups),
        "json_schema_valid_outputs": valid,
        "summary": summary, "rows": rows,
        "reference_digests": {f"{k[0]}-{k[1]}": v["digest"] for k, v in ref.items()},
        "config": {"VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
                   "async_scheduling": async_sched, "prefix_caching": False, "max_num_seqs": 64,
                   "temperature": 0.0, "backend": "guidance"},
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__,
                     "llguidance": sh("pip show llguidance | grep Version"),
                     "python": platform.python_version()},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "cpu": sh("grep -m1 'model name' /proc/cpuinfo"),
                     "nproc": os.cpu_count(), "sched_affinity": len(os.sched_getaffinity(0)),
                     "cgroup_cpu_max": sh("cat /sys/fs/cgroup/cpu.max"),
                     "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")},
    }
    json.dump(out, open(os.path.join(args.out, "measurements.json"), "w"), indent=1)
    json.dump({f"{k[0]}-{k[1]}": v["texts"] for k, v in ref.items()},
              open(os.path.join(args.out, "outputs.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("outputs_token_identical_all_runs", "mismatched_runs", "usable_schemas", "json_schema_valid_outputs", "summary", "config", "versions", "hardware")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
