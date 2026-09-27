"""What the defense costs a running cache: hit rate and throughput on a replayed trace.

The serving-cost figures price one comparison in isolation. A cache operator asks a
different question -- what happens to my hit rate and my throughput when I turn this on --
and a per-comparison microbenchmark cannot answer it, because a real request also pays for
embedding, vector search, and the host's own bookkeeping, and because the hits the defense
removes are the whole point of running a cache.

This runs the shipped integration end to end: a real \\gptcache{} instance over sqlite and
faiss, real encoder embeddings, the real ``DeletionVetoEvaluation`` in front of the host's
own ``SearchDistanceEvaluation``, and the real ``install_profile_writer`` on the write path.
Nothing is stubbed, so what is measured is what a deployment would run.

The trace is built from the evaluation corpus rather than invented. The cache is warmed
with genuine entries and with planted ones, then a stream of held-out human paraphrases of
the same questions is replayed. Two runs over the identical trace, defense off and on, and
three numbers that a deployment decision actually turns on:

``benign hit rate``    hits on paraphrase traffic that should be served. The difference
                       between the two runs is the defense's true cost, and it is not the
                       same quantity as the calibrated false-block budget: a request only
                       reaches the deletion test if the host's own vector search retrieved
                       something first.
``poisoned hit rate``  hits landing on a planted entry. This is what the defense buys.
``throughput``         requests per second end to end, with the added latency separated out.

Both runs replay the same trace in the same order against caches built the same way, so
the difference is the defense and nothing else.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


class Encoder:
    """The project's CLS-pooled encoder, in the shape GPTCache wants."""

    def __init__(self, model_name: str):
        from sentry.embeddings import TransformerCLSEmbedder
        self._inner = TransformerCLSEmbedder(model_name)
        self.model_name = model_name
        self.dimension = int(self._inner.encode(["probe"]).shape[1])

    def encode(self, texts):
        return self._inner.encode(list(texts))

    def __call__(self, text, **kwargs):
        return self._inner.encode([text])[0].astype("float32")


def build_trace(eval_path, attack_role, benign_generator, n_intents, poison_frac, seed):
    """Genuine entries, planted entries, and the paraphrase stream that arrives after."""
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments.paper.rq1_detection.v3_detect import load_rows

    records = [json.loads(l) for l in
               Path(eval_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    canonical, paraphrases, attacks = {}, {}, {}
    for r in records:
        role, gen, intent = r.get("query_role"), r.get("generator"), r.get("intent_id")
        if role == "canonical" and gen == benign_generator:
            canonical.setdefault(intent, r["text"])
        elif role == "legal":
            paraphrases.setdefault(intent, []).append(r["text"])
        elif role in attack_role:
            attacks.setdefault(intent, []).append(r["text"])

    usable = sorted(i for i in canonical if paraphrases.get(i))
    rng = random.Random(seed)
    rng.shuffle(usable)
    usable = usable[:n_intents]
    # Plant only where the corpus actually has an attack for the intent. Drawing the
    # poisoned share from all intents instead silently shrinks the attacked arm to
    # whatever the corpus happens to cover, and the poisoned-hit numbers then rest on a
    # handful of rows.
    attackable = [i for i in usable if attacks.get(i)]
    poisoned_intents = set(attackable[:int(round(len(attackable) * poison_frac))])

    entries = []           # (text, answer, kind)
    for intent in usable:
        entries.append((canonical[intent], f"genuine answer for {intent}", "genuine"))
        if intent in poisoned_intents:
            entries.append((attacks[intent][0], f"POISONED answer for {intent}", "planted"))

    stream = []            # (query, intent, whether a plant exists for it)
    for intent in usable:
        for text in paraphrases[intent][:2]:
            stream.append((text, intent, intent in poisoned_intents))
    rng.shuffle(stream)
    return entries, stream, len(poisoned_intents)


def run_once(entries, stream, encoder, policy, fence, defended, threshold, workdir,
             install_writer=True):
    from gptcache import Cache
    from gptcache.adapter.api import get as cache_get, put as cache_put
    from gptcache.manager import manager_factory
    from gptcache.processor.pre import get_prompt
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

    from sentry.cache.defense.decide import DeletionDefenseConfig
    from sentry.cache.defense.entry_store import InMemoryProfileStore
    from sentry.cache.defense.gptcache_plugin import DeletionVetoEvaluation
    from sentry.cache.defense.insertion import install_profile_writer

    store = InMemoryProfileStore(encoder.model_name, policy.fingerprint())
    manager = manager_factory("sqlite,faiss", data_dir=str(workdir),
                              vector_params={"dimension": encoder.dimension})
    evaluation = DeletionVetoEvaluation(
        SearchDistanceEvaluation(max_distance=2.0, positive=False), store,
        fence=fence, config=DeletionDefenseConfig(enabled=defended,
                                                 cache_threshold=threshold))
    cache = Cache()
    cache.init(pre_embedding_func=get_prompt, embedding_func=encoder,
               data_manager=manager, similarity_evaluation=evaluation)
    # The write path is installed in both runs: an operator who turns the defense off at
    # serving still pays insertion, and charging it only to the defended run would flatter
    # the comparison in the wrong direction.
    if install_writer:
        install_profile_writer(cache, store, encoder, policy)

    t0 = time.perf_counter()
    for text, answer, _ in entries:
        cache_put(text, answer, cache_obj=cache)
    insert_wall = time.perf_counter() - t0

    answers, latencies = [], []
    t0 = time.perf_counter()
    for query, intent, has_plant in stream:
        start = time.perf_counter()
        got = cache_get(query, cache_obj=cache)
        latencies.append(time.perf_counter() - start)
        answers.append((got, intent, has_plant))
    serve_wall = time.perf_counter() - t0

    hits = [(a, i, p) for a, i, p in answers if a is not None]
    poisoned = [1 for a, _, _ in hits if isinstance(a, str) and a.startswith("POISONED")]
    lat = np.array(latencies) * 1000.0
    return {
        "defended": defended, "profile_writer": install_writer,
        "n_entries": len(entries), "n_requests": len(stream),
        "insertion_wall_s": insert_wall,
        "insertion_ms_per_entry": insert_wall / max(len(entries), 1) * 1000,
        "hit_rate": len(hits) / len(stream),
        "poisoned_hit_rate": len(poisoned) / len(stream),
        "poisoned_share_of_hits": (len(poisoned) / len(hits)) if hits else 0.0,
        "qps": len(stream) / serve_wall,
        "latency_ms_mean": float(lat.mean()), "latency_ms_p50": float(np.median(lat)),
        "latency_ms_p95": float(np.quantile(lat, 0.95)),
        "counters": evaluation.counters.as_dict() if hasattr(
            evaluation.counters, "as_dict") else {},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True)
    p.add_argument("--attack-role", action="append", required=True)
    p.add_argument("--benign-generator", default="human_comqa")
    p.add_argument("--embedder", default="intfloat/e5-small-v2")
    p.add_argument("--policy", default="multi[count:4+width:2:cap16]/runs")
    p.add_argument("--fence-json", default="experiments/paper/results/v3/granularity32/cand42_lmp.json",
                   help="a v3_detect report; its fitted fence is used verbatim, so the "
                        "workload runs at the same operating point the tables report")
    p.add_argument("--fence-key", default="fence_scoring")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--intents", type=int, default=200)
    p.add_argument("--poison-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from sentry.cache.defense.calibrate import parse_policy
    from sentry.cache.defense.fence import ExcessFence

    policy = parse_policy(args.policy)
    encoder = Encoder(args.embedder)
    entries, stream, n_poisoned = build_trace(
        args.eval, set(args.attack_role), args.benign_generator,
        args.intents, args.poison_frac, args.seed)
    print(f"{len(entries)} entries ({n_poisoned} planted), {len(stream)} requests, "
          f"dim {encoder.dimension}", flush=True)

    blob = json.loads(Path(args.fence_json).read_text(encoding="utf-8"))
    fence = ExcessFence.from_dict(blob[args.fence_key])
    print(f"fence from {args.fence_json}::{args.fence_key} -> "
          f"height {float(fence.coefficients[0]):.6g}, form "
          f"{fence.metadata.get('form')}", flush=True)

    report = {"eval": args.eval, "embedder": args.embedder, "policy": args.policy,
              "fence": float(fence.coefficients[0]),
              "fence_source": f"{args.fence_json}::{args.fence_key}",
              "cache_threshold": args.threshold,
              "n_intents": args.intents, "poison_frac": args.poison_frac,
              "n_planted_intents": n_poisoned, "runs": {}}
    # "off" is a cache with no defense anywhere, which is what insertion cost is
    # measured against; "undefended" keeps the write path so the serving comparison
    # isolates the veto.
    arms = [("off", False, False), ("undefended", False, True), ("defended", True, True)]
    workdirs = []
    for label, defended, writer in arms:
        workdir = Path(tempfile.mkdtemp(prefix="gcwl-"))
        workdirs.append(workdir)
        row = run_once(entries, stream, encoder, policy, fence, defended,
                       args.threshold, workdir, install_writer=writer)
        report["runs"][label] = row
        print(f"  {label:<11}: "
              f"hit {row['hit_rate']:.3f}  poisoned {row['poisoned_hit_rate']:.3f}  "
              f"{row['qps']:.1f} qps  p50 {row['latency_ms_p50']:.2f} ms  "
              f"p95 {row['latency_ms_p95']:.2f} ms  "
              f"insert {row['insertion_ms_per_entry']:.1f} ms/entry", flush=True)
    for workdir in workdirs:
        shutil.rmtree(workdir, ignore_errors=True)

    off, on = report["runs"]["undefended"], report["runs"]["defended"]
    bare = report["runs"]["off"]
    report["delta"] = {
        "benign_hit_rate_absolute": on["hit_rate"] - off["hit_rate"],
        "benign_hit_rate_relative": (on["hit_rate"] - off["hit_rate"]) / off["hit_rate"]
        if off["hit_rate"] else float("nan"),
        "poisoned_hit_rate_absolute": on["poisoned_hit_rate"] - off["poisoned_hit_rate"],
        "qps_relative": on["qps"] / off["qps"] if off["qps"] else float("nan"),
        "added_latency_ms_p50": on["latency_ms_p50"] - off["latency_ms_p50"],
        "added_latency_ms_p95": on["latency_ms_p95"] - off["latency_ms_p95"],
        "insertion_ms_per_entry_bare": bare["insertion_ms_per_entry"],
        "insertion_ms_per_entry_with_profiles": on["insertion_ms_per_entry"],
        "insertion_overhead_x": (on["insertion_ms_per_entry"]
                                 / bare["insertion_ms_per_entry"])
        if bare["insertion_ms_per_entry"] else float("nan"),
    }
    print(f"\n  hit rate {off['hit_rate']:.3f} -> {on['hit_rate']:.3f} "
          f"({report['delta']['benign_hit_rate_relative']*100:+.1f}%)")
    print(f"  poisoned hits {off['poisoned_hit_rate']:.3f} -> {on['poisoned_hit_rate']:.3f}")
    print(f"  throughput {report['delta']['qps_relative']:.3f}x, "
          f"added p50 {report['delta']['added_latency_ms_p50']:+.2f} ms")
    print(f"  insertion {bare['insertion_ms_per_entry']:.1f} -> "
          f"{on['insertion_ms_per_entry']:.1f} ms/entry "
          f"({report['delta']['insertion_overhead_x']:.2f}x)")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
