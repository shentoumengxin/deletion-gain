import json, os, shutil, hashlib
from sentry.artifacts import data_path
D = str(data_path("datasets", "answer_check_20260909"))
out = os.path.join(D, "sets", "scored")
os.makedirs(out, exist_ok=True)
for name in ("unseen-wrappers", "non-echo", "surface-form", "projection"):
    shutil.copy(os.path.join(D, "sets", name + ".jsonl"), os.path.join(out, name + ".jsonl"))

def split_of(intent_id, seed=20260909, frac=0.5):
    v = int(hashlib.sha256(("instruction-benign:%d:%s" % (seed, intent_id)).encode()).hexdigest()[:16], 16) / 2**64
    return "calibration" if v < frac else "test"

rows = []
for l in open(os.path.join(D, "prompts", "prompts_rq5.jsonl")):
    d = json.loads(l)
    rows.append({"prompt": d["prompt"], "text": d["prompt"], "anchor": d["anchor"],
                 "intent_id": d["intent_id"], "corpus": d["corpus"], "set": d["set"],
                 "template": d["template"], "position": d["position"],
                 "condition": d["condition"], "split": split_of(d["intent_id"]),
                 "kind": "qualifier", "malicious": False,
                 "base_text": d["base_text"], "clause": d["clause"],
                 "sample_id": hashlib.sha256(d["prompt"].encode()).hexdigest()[:24]})
with open(os.path.join(out, "rq5.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rq5", len(rows))
for f in sorted(os.listdir(out)):
    print(f, sum(1 for _ in open(os.path.join(out, f))))
