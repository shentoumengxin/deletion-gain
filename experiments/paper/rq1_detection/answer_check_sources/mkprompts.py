import json, os
from sentry.artifacts import data_path
os.chdir(data_path("datasets", "answer_check_20260909"))
os.makedirs("prompts", exist_ok=True)
for name in ("unseen-wrappers", "non-echo", "surface-form"):
    seen, rows = set(), []
    for l in open("sets/%s.jsonl" % name):
        d = json.loads(l); p = d["prompt"]
        if p in seen: continue
        seen.add(p)
        rows.append({"prompt": p, "set": d["set"], "corpus": d.get("corpus"),
                     "intent_id": d.get("intent_id"), "record_id": d.get("record_id")})
    fn = "prompts/prompts_%s.jsonl" % name.replace("-", "_")
    with open(fn, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))
seen, rows = set(), []
for l in open("sets/non-echo.jsonl"):
    d = json.loads(l); c = d.get("control_prompt")
    if c and c not in seen:
        seen.add(c)
        rows.append({"prompt": c, "set": "non_echo_control", "intent_id": d.get("intent_id"), "corpus": d.get("corpus")})
with open("prompts/prompts_non_echo_control.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("non_echo_control", len(rows))
