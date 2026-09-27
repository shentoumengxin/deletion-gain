"""The cache-space hardening family, measured instead of asserted.

The evaluation claims one representative per defense family and then leaves the hardening
row empty. This fills it. Hardening does not score an entry -- it perturbs the space the
matching happens in, so an attacker who cannot see the transform cannot solve for a vector
that collides. Its "block" is therefore a *retrieval* event: the attack no longer reaches
the entry it was aimed at.

That makes it directly comparable to every other row once the operating point is stated the
same way. Fix the retrieval threshold under the salt so that exactly ``budget`` of benign
entries lose their cache hit -- the same false-positive currency every other method is
charged in -- and report the share of attacks that lose theirs. The score used for AUC is
``-cos_R(entry, anchor)``, higher meaning more suspicious, matching the sign convention of
the cosine row.

Two transforms, and they fail for different reasons:

**Orthogonal ``R``** -- the natural reading of "hide the geometry behind a secret matrix".
Orthogonal maps preserve inner products, so ``cos(Rx, Ry) = cos(x, y)`` exactly. The
acceptance region is the *same set of texts*; the attacker never needs to recover ``R``.
This is an identity, not an empirical finding, and the run confirms it to floating point so
it cannot be waved away.

**Compressive ``R``** (a random ``k x D`` projection, ``k < D``). This one does move the
geometry and has to be measured. The question is not whether it perturbs cosines -- it does
-- but whether it perturbs *attacks* more than *benign paraphrases*. If both move together,
the attack still lands inside the acceptance region and the operator has bought only
calibration noise and a coarser cache.

The reason to expect little is worth stating in advance: these attacks are not solved
against the encoder's geometry. A payload attached to a genuine question collides because it
genuinely shares the question's words, and it would collide under any reasonable encoder.
Salting defeats an attacker who solves for a specific geometry, not one who writes a
sentence that means nearly the same thing. \\gcg{} is the exception that tests this, since
its suffix *is* optimized against the encoder.

Writes per-row scores in the shape ``rq1_operating_points.py --extra`` consumes, so the
hardening row is read at the same budgets, under the same intent-grouped protocol, as
every other row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def orthogonal_salt(dimension: int, seed: int) -> np.ndarray:
    """A secret rotation of the embedding space. Preserves every inner product."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return q


def compressive_salt(dimension: int, width: int, seed: int) -> np.ndarray:
    """A secret ``width x D`` random projection. Changes the geometry; the question is how."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(width, dimension)) / np.sqrt(width)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True)
    p.add_argument("--attack-role", action="append", required=True)
    p.add_argument("--benign-generator", action="append", required=True)
    p.add_argument("--embedder", default="intfloat/e5-small-v2")
    p.add_argument("--widths", default="256,128,64,32")
    p.add_argument("--floor", type=float, default=0.90,
                   help="the cache's own retrieval threshold, for the retrievability report")
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--out", required=True)
    p.add_argument("--summary", default="")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    from experiments.paper.rq1_detection.v3_detect import load_rows
    from sentry.embeddings import TransformerCLSEmbedder

    rows, _ = load_rows(args.eval, args.attack_role, args.benign_generator)
    print(f"{sum(r['arm'] == 'attack' for r in rows)} attack / "
          f"{sum(r['arm'] == 'genuine' for r in rows)} benign", flush=True)

    embedder = TransformerCLSEmbedder(args.embedder)
    pool: dict[str, int] = {}
    for row in rows:
        pool.setdefault(row["text"], len(pool))
        pool.setdefault(row["anchor"], len(pool))
    texts = [None] * len(pool)
    for text, i in pool.items():
        texts[i] = text
    start = time.perf_counter()
    matrix = np.concatenate(
        [embedder.encode(texts[i:i + 256]) for i in range(0, len(texts), 256)], axis=0)
    print(f"  encoded {len(texts)} texts in {time.perf_counter() - start:.0f}s", flush=True)

    entry = np.array([pool[r["text"]] for r in rows])
    anchor = np.array([pool[r["anchor"]] for r in rows])
    is_attack = np.array([r["arm"] == "attack" for r in rows])
    intents = [r["intent_id"] for r in rows]
    dimension = matrix.shape[1]

    salts = {"no salt": None,
             "orthogonal (secret rotation)": orthogonal_salt(dimension, args.seed)}
    for w in (int(x) for x in args.widths.split(",") if x):
        salts[f"compressive k={w}"] = compressive_salt(dimension, w, args.seed + w)

    per_row, summary = {}, {}
    baseline_cos = None
    for name, R in salts.items():
        projected = unit_rows(matrix if R is None else matrix @ R.T)
        cos = np.einsum("ij,ij->i", projected[entry], projected[anchor])
        if baseline_cos is None:
            baseline_cos = cos
        # higher = more suspicious, so the sign matches the cosine row elsewhere
        per_row[f"salting, {name}"] = {
            "attack": (-cos[is_attack]).tolist(),
            "benign": (-cos[~is_attack]).tolist(),
            "benign_intents": [i for i, a in zip(intents, is_attack) if not a],
            # Carried so the attack arm can be restricted to the plants the judge
            # confirmed poisoned the victim -- the Succ. column every other method
            # reports. Without it the row has three blanks that look like a limitation
            # of the method rather than of the dump.
            "attack_record_ids": [r["record_id"] for r, a in zip(rows, is_attack) if a],
        }
        summary[name] = {
            "attack_retrievable": float((cos[is_attack] >= args.floor).mean()),
            "benign_retrievable": float((cos[~is_attack] >= args.floor).mean()),
            "median_attack_cos": float(np.median(cos[is_attack])),
            "median_benign_cos": float(np.median(cos[~is_attack])),
            "max_abs_cos_shift_vs_unsalted": float(np.max(np.abs(cos - baseline_cos))),
        }
        print(f"  {name:<30} attack retrievable {summary[name]['attack_retrievable']:.4f}  "
              f"benign {summary[name]['benign_retrievable']:.4f}  "
              f"max|dcos| {summary[name]['max_abs_cos_shift_vs_unsalted']:.2e}", flush=True)

    Path(args.out).write_text(json.dumps(per_row), encoding="utf-8")
    print(f"wrote {args.out}")
    if args.summary:
        Path(args.summary).write_text(json.dumps(
            {"embedder": args.embedder, "dimension": dimension, "floor": args.floor,
             "seed": args.seed, "salts": summary}, indent=2), encoding="utf-8")
        print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
