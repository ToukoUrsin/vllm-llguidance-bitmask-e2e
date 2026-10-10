set -ex
apt-get update -qq && apt-get install -y -qq gcc g++ > /dev/null
pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install -q --no-cache-dir llguidance==1.9.1 transformers pytest numpy tokenizers jinja2
SP=$(python -c "import llguidance,os;print(os.path.dirname(llguidance.__file__))")
OUT=out; mkdir -p $OUT
diff $SP/torch.py torch_orig.py && echo "wheel torch.py == upstream main torch.py" | tee $OUT/wheel-equal.txt
mkdir -p t && cp test_hf.py test_bitmask.py __init__.py t/
python -m pytest -q -p no:logging t/test_hf.py t/test_bitmask.py 2>&1 | tail -6 | tee $OUT/pytest-orig.txt || true
cp torch_patched.py $SP/torch.py
python -m pytest -q -p no:logging -rA t/test_hf.py t/test_bitmask.py 2>&1 | grep -E "PASSED|FAILED|SKIPPED|ERROR|passed|failed" | tee $OUT/pytest-patched.txt
python - <<'P' 2>&1 | tee $OUT/micro.txt
import time, torch, importlib.util, llguidance, llguidance.hf
from llguidance import LLMatcher
from transformers import AutoTokenizer
t = llguidance.hf.from_tokenizer(AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"))
def load(p):
    s = importlib.util.spec_from_file_location("m"+p[:4], p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
old, new = load("torch_orig.py"), load("torch_patched.py")
m = LLMatcher(t, LLMatcher.grammar_from_json_schema({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}))
m.consume_tokens(t.tokenize_str('{"a": "x'))
bm = old.allocate_token_bitmask(64, t.vocab_size)
a = bm.clone(); b = bm.clone()
for i in range(64):
    old.fill_next_token_bitmask(m, a, i); new.fill_next_token_bitmask(m, b, i)
assert torch.equal(a, b)
N = 20000
res = {"old": [], "new": []}
for rep in range(5):
    for name, f in (("old", old.fill_next_token_bitmask), ("new", new.fill_next_token_bitmask))[:: 1 if rep % 2 == 0 else -1]:
        t0 = time.perf_counter_ns()
        for i in range(N):
            f(m, bm, i & 63)
        res[name].append((time.perf_counter_ns() - t0) / N / 1000)
import statistics
print({k: round(statistics.median(v), 3) for k, v in res.items()}, "us/call (mid-string JSON state, Qwen2.5 vocab, 5 alternating repeats)")
P
