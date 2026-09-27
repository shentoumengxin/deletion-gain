#!/usr/bin/env python
"""B5 reference attack: strengthened fork of the CacheAttack GCG generator.

The third-party reference attack for the defense evaluation. The
vendored submodule ``CacheAttack/attack/cache_attack.py`` is imported and
subclassed — never modified — and hardened in three ways that matter for
measuring attack COST rather than binary attack success:

1.  **No early stop.** Upstream ``optimize_suffix`` returns as soon as the true
    cosine reaches ``target_sim`` (0.88) and returns ``None`` on budget
    exhaustion. Here the loop always consumes the full step budget and always
    returns the best suffix seen, so cost is comparable across tasks and there
    is no ``None`` failure sentinel.
2.  **Per-step cost trace.** Every step logs the true cosine to the target,
    the combined objective score, the perplexity of the current candidate
    (``_gcg_step`` now returns it), and the suffix token length.
3.  **Checkpoint sweep on ONE trajectory.** The first step at which the
    running-best cosine crosses each target (default 0.88/0.92/0.95/0.97) is
    recorded along a single optimization — cost-to-reach per target — instead
    of four separate early-stopped runs.

F1/F2 variants: F1 drops the perplexity penalty from the scorer
(``--lambda-ppl 0.0``, the default here; ppl is still logged), F2 keeps it
(``--lambda-ppl 0.04`` — the default in CacheAttack's hijack evaluation
driver; the vendored ``CacheAttackGenerator`` constructor default is 0.05).

Surrogate/target models follow ``CacheAttack/evaluation/hijack.py``: the
surrogate is ``BAAI/bge-small-en-v1.5`` (default ``--embed-model``); the
defense-side target ``intfloat/e5-small-v2`` is not needed here — this is the
attacker side, cosine mode only.

CLI: JSONL tasks in -> JSONL results + JSONL cost traces out, with
``--shard-index/--shard-count`` for array jobs. ``--selftest`` runs offline:
mocked-generator checks always, plus a real-model smoke test only when
all-MiniLM-L6-v2 is already in the local HF cache (never downloaded).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

# Vendored third-party submodule (git submodule, unmodified) — imported, not copied.
from CacheAttack.attack.cache_attack import CacheAttackGenerator, device  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_TARGETS = (0.88, 0.92, 0.95, 0.97)


def _ckpt_key(target: float) -> str:
    return f"{target:g}"


class ReferenceAttackGenerator(CacheAttackGenerator):
    """Strengthened CacheAttackGenerator: fixed budget, traced, checkpointed.

    Cosine mode only (the hijack evaluation's setting for standard semantic
    caches). ``optimize_suffix`` is reimplemented (no early stop, per-step
    trace, checkpoint sweep) and ``_gcg_step`` is wrapped to also return the
    chosen candidate's perplexity; everything else is upstream verbatim.
    """

    def __init__(self, *args, checkpoint_targets=DEFAULT_CHECKPOINT_TARGETS, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_targets = tuple(float(t) for t in checkpoint_targets)

    # ------------------------------------------------------------------ seams
    # Small helpers so tests/selftest can replay a scripted trajectory without
    # touching a tokenizer or a model.

    def _prepare(self, p_src: str, p_v: str):
        """Tokenise the payload and pre-compute the target embedding."""
        src_ids = self.embed_tokenizer(
            p_src, return_tensors="pt", add_special_tokens=False
        ).input_ids[0].to(device)
        tgt_enc = self.embed_tokenizer(p_v, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            tgt_emb = self._get_embedding(tgt_enc["input_ids"], tgt_enc["attention_mask"])
        return src_ids, tgt_emb

    def _true_cosine(self, p_src: str, suffix_ids: torch.Tensor, tgt_emb: torch.Tensor) -> float:
        """Re-embed payload + prefix + suffix and return cosine to the target."""
        suffix_text = self.embed_tokenizer.decode(suffix_ids, skip_special_tokens=True)
        enc = self.embed_tokenizer(
            f"{p_src} {self.suffix_prefix}{suffix_text}",
            return_tensors="pt",
            truncation=True,
        ).to(device)
        with torch.no_grad():
            emb = self._get_embedding(enc["input_ids"], enc["attention_mask"])
        return float((emb * tgt_emb).sum().item())

    # -------------------------------------------------------------- overrides

    def _gcg_step(self, src_ids, suffix_ids, tgt_repr, top_k, batch_size, s_src):
        """Upstream GCG step, plus the chosen candidate's perplexity.

        Provenance: delegates to ``CacheAttackGenerator._gcg_step`` verbatim.
        The upstream batch loop already computes each candidate's ppl for the
        combined score but discards it; the winner's ppl is recomputed once
        here (deterministic LM forward — one extra call per step, negligible
        next to the batch_size evaluations). Returns a 3-tuple, so the
        upstream ``run_dynamic_attack`` (2-tuple unpacking) is unsupported in
        this fork.
        """
        best_ids, best_score = super()._gcg_step(
            src_ids, suffix_ids, tgt_repr, top_k, batch_size, s_src
        )
        suffix_text = self.embed_tokenizer.decode(best_ids, skip_special_tokens=True)
        ppl = self._compute_ppl(f"{s_src} {self.suffix_prefix}{suffix_text}")
        return best_ids, best_score, ppl

    def optimize_suffix(
        self,
        p_src: str,
        p_v: str,
        suffix_len: int,
        steps: int = 400,
        batch_size: int = 64,
        top_k: int = 64,
        checkpoint_targets=None,
        verbose: bool = False,
    ) -> dict:
        """Fixed-budget GCG. NEVER early-stops, NEVER returns None.

        Returns a dict with ``suffix`` (best seen, no prefix), ``final_cosine``
        (cosine of that best suffix), ``checkpoints`` (target -> first step at
        which the running-best cosine crosses it, or None), and ``cost_trace``
        (one row per step).
        """
        targets = tuple(
            self.checkpoint_targets if checkpoint_targets is None else checkpoint_targets
        )
        src_ids, tgt_emb = self._prepare(p_src, p_v)
        tgt_repr = self._lsh_hash_soft(tgt_emb) if self.mode == "lsh" else tgt_emb

        suffix_ids = torch.randint(0, self.vocab_size, (suffix_len,), device=device)
        best_cos = float("-inf")
        best_suffix_ids = suffix_ids.clone()
        first_cross = {_ckpt_key(t): None for t in targets}
        trace = []

        for step in range(steps):
            suffix_ids, score, ppl = self._gcg_step(
                src_ids, suffix_ids, tgt_repr, top_k, batch_size, p_src
            )
            true_cos = self._true_cosine(p_src, suffix_ids, tgt_emb)
            trace.append({
                "step": step,
                "true_cosine": round(true_cos, 6),
                "combined_score": round(float(score), 6),
                "ppl": round(float(ppl), 4),
                "suffix_token_len": int(suffix_ids.numel()),
            })
            if true_cos > best_cos:
                best_cos = true_cos
                best_suffix_ids = suffix_ids.clone()
            for t in targets:
                key = _ckpt_key(t)
                if first_cross[key] is None and best_cos >= t:
                    first_cross[key] = step
            if verbose and step % 50 == 0:
                logger.info(
                    "  step=%4d cos=%.4f best=%.4f L=%d", step, true_cos, best_cos, suffix_len
                )

        return {
            "suffix": self.embed_tokenizer.decode(best_suffix_ids, skip_special_tokens=True),
            "final_cosine": round(best_cos, 6),
            "checkpoints": first_cross,
            "cost_trace": trace,
        }


# --------------------------------------------------------------------------- driver


def load_tasks(tasks_path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(tasks_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def shard_tasks(tasks: list[dict], shard_index: int = 0, shard_count: int = 1) -> list[dict]:
    if not (0 <= shard_index < shard_count):
        raise ValueError(f"need 0 <= shard-index < shard-count, got {shard_index}/{shard_count}")
    return tasks[shard_index::shard_count]


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    Path(path).write_text(text + ("\n" if rows else ""), encoding="utf-8")


def _append_jsonl(path: str | Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_tasks(
    tasks: list[dict],
    generator: ReferenceAttackGenerator,
    *,
    steps: int,
    batch_size: int,
    top_k: int,
    suffix_len: int,
    embed_model_name: str,
    lambda_ppl: float,
    checkpoint_targets=None,
    trace_path: str | Path | None = None,
    output_path: str | Path | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Optimize one suffix per task. Cost traces go to ``trace_path`` when
    given (kept out of the main output to keep it slim), else embedded in the
    main rows under ``cost_trace``. When ``output_path`` is given, each row is
    also appended to it as soon as its task finishes, so a multi-hour run
    loses nothing to a crash; tasks already present there are skipped."""
    done_keys: set[tuple[str, str]] = set()
    if output_path is not None and Path(output_path).exists():
        with Path(output_path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    prior = json.loads(line)
                    done_keys.add((prior["target_question"], prior["payload"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    rows, trace_rows = [], []
    for task in tasks:
        p_src, p_v = task["payload"], task["target_question"]
        if (p_v, p_src) in done_keys:
            logger.info("skip (already in output): %s", p_v[:60])
            continue
        L = int(task.get("suffix_len", suffix_len))
        result = generator.optimize_suffix(
            p_src=p_src,
            p_v=p_v,
            suffix_len=L,
            steps=steps,
            batch_size=batch_size,
            top_k=top_k,
            checkpoint_targets=checkpoint_targets,
            verbose=verbose,
        )
        row = {
            "target_question": p_v,
            "payload": p_src,
            "attack_text": f"{p_src} {generator.suffix_prefix}{result['suffix']}",
            "suffix": result["suffix"],
            "suffix_prefix": generator.suffix_prefix,
            "final_cosine": result["final_cosine"],
            "checkpoints": result["checkpoints"],
            "lambda_ppl": lambda_ppl,
            "embed_model": embed_model_name,
            "steps": steps,
        }
        if trace_path is None:
            row["cost_trace"] = result["cost_trace"]
        else:
            trace_row = {
                "target_question": p_v,
                "payload": p_src,
                "cost_trace": result["cost_trace"],
            }
            trace_rows.append(trace_row)
            if trace_path is not None:
                _append_jsonl(trace_path, trace_row)
        rows.append(row)
        if output_path is not None:
            _append_jsonl(output_path, row)
        logger.info(
            "task done: cos=%.4f checkpoints=%s q=%s",
            result["final_cosine"], result["checkpoints"], p_v[:60],
        )
    if trace_path is not None and output_path is None:
        _write_jsonl(trace_path, trace_rows)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="offline checks, no downloads")
    parser.add_argument("--tasks", help="JSONL of {target_question, payload, optional suffix_len}")
    parser.add_argument("--output", default="reference_attack.jsonl")
    parser.add_argument("--trace-output", default=None,
                        help="JSONL for per-step cost traces (kept out of --output)")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--lm-model", default="gpt2")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--suffix-len", type=int, default=20)
    parser.add_argument("--lambda-ppl", type=float, default=0.0,
                        help="F1=0.0 (no ppl penalty, still logged); F2=0.04 (hijack default)")
    parser.add_argument("--checkpoint-targets", default="0.88,0.92,0.95,0.97",
                        help="comma-separated cosine targets for the single-trajectory sweep")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser


# --------------------------------------------------------------------------- selftest


class _MockTokenizer:
    def decode(self, ids, skip_special_tokens=True) -> str:
        return " ".join(str(int(i)) for i in ids)


def build_mock_generator(
    cosines: list[float],
    scores: list[float] | None = None,
    ppls: list[float] | None = None,
    lambda_ppl: float = 0.0,
    checkpoint_targets=DEFAULT_CHECKPOINT_TARGETS,
) -> ReferenceAttackGenerator:
    """Model-free ReferenceAttackGenerator for selftest and unit tests.

    Bypasses ``__init__`` (no model downloads); ``_prepare`` / ``_gcg_step`` /
    ``_true_cosine`` replay the scripted trajectory, one entry per step.
    """
    n = len(cosines)
    scores = scores if scores is not None else [0.0] * n
    ppls = ppls if ppls is not None else [1.0] * n
    assert len(scores) == n and len(ppls) == n

    gen = object.__new__(ReferenceAttackGenerator)
    gen.mode = "cos"
    gen.lambda_ppl = lambda_ppl
    gen.suffix_prefix = "Neglect: "
    gen.vocab_size = 50_000
    gen.embed_tokenizer = _MockTokenizer()
    gen.checkpoint_targets = tuple(float(t) for t in checkpoint_targets)
    state = {"i": 0}

    def _prepare(p_src, p_v):
        return torch.zeros(2, dtype=torch.long, device=device), torch.ones(4, device=device)

    def _gcg_step(src_ids, suffix_ids, tgt_repr, top_k, batch_size, s_src):
        i = state["i"]
        return suffix_ids, scores[i], ppls[i]

    def _true_cosine(p_src, suffix_ids, tgt_emb):
        i = state["i"]
        state["i"] = i + 1
        return cosines[i]

    gen._prepare = _prepare
    gen._gcg_step = _gcg_step
    gen._true_cosine = _true_cosine
    return gen


def _cached_snapshot(repo_id: str) -> str | None:
    """Local HF cache path for repo_id, or None if not already cached. Never downloads."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return None


def _selftest() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name, fn):
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as e:  # noqa: BLE001 - selftest reports, not crashes
            checks.append((name, False, repr(e)))

    def t_no_early_stop():
        cos = [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.87, 0.93, 0.91, 0.89]
        gen = build_mock_generator(cos)
        out = gen.optimize_suffix("payload", "target", suffix_len=5, steps=len(cos))
        assert len(out["cost_trace"]) == len(cos), "must consume the full budget"
        assert out["checkpoints"]["0.88"] == 5, out["checkpoints"]
        assert out["final_cosine"] == 0.93 and out["suffix"] is not None

    def t_checkpoints():
        cos = [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.87, 0.93, 0.91, 0.89]
        out = build_mock_generator(cos).optimize_suffix("p", "t", suffix_len=5, steps=len(cos))
        assert out["checkpoints"] == {"0.88": 5, "0.92": 7, "0.95": None, "0.97": None}
        flat = [0.3, 0.4, 0.35]
        out2 = build_mock_generator(flat).optimize_suffix("p", "t", suffix_len=5, steps=3)
        assert all(v is None for v in out2["checkpoints"].values())
        assert out2["suffix"] is not None and out2["final_cosine"] == 0.4

    def t_trace_fields():
        out = build_mock_generator([0.1, 0.2], scores=[1.0, 2.0], ppls=[10.0, 20.0]).optimize_suffix(
            "p", "t", suffix_len=5, steps=2
        )
        for i, row in enumerate(out["cost_trace"]):
            assert set(row) == {"step", "true_cosine", "combined_score", "ppl", "suffix_token_len"}
            assert row["step"] == i and row["suffix_token_len"] == 5
        assert out["cost_trace"][1]["ppl"] == 20.0

    def t_lambda_plumbing():
        import CacheAttack.attack.cache_attack as upstream

        seen = []
        orig = upstream.CacheAttackGenerator._gcg_step

        def spy(self, *a, **k):
            seen.append(self.lambda_ppl)
            return torch.zeros(3, dtype=torch.long, device=device), 0.0

        upstream.CacheAttackGenerator._gcg_step = spy
        try:
            for lam in (0.0, 0.04):
                gen = object.__new__(ReferenceAttackGenerator)
                gen.lambda_ppl = lam
                gen.suffix_prefix = "Neglect: "
                gen.embed_tokenizer = _MockTokenizer()
                gen._compute_ppl = lambda text: 42.0
                ids, score, ppl = gen._gcg_step(None, None, None, 8, 4, "src")
                assert ppl == 42.0 and score == 0.0
        finally:
            upstream.CacheAttackGenerator._gcg_step = orig
        assert seen == [0.0, 0.04], f"lambda_ppl must reach the scorer, saw {seen}"

    for name, fn in [
        ("no_early_stop", t_no_early_stop),
        ("checkpoints_first_crossing", t_checkpoints),
        ("cost_trace_fields", t_trace_fields),
        ("lambda_ppl_plumbing", t_lambda_plumbing),
    ]:
        check(name, fn)

    # Optional real-model smoke test: only when already in the local HF cache.
    snap = _cached_snapshot("sentence-transformers/all-MiniLM-L6-v2")
    if snap is None:
        print("SKIP real_model_smoke: all-MiniLM-L6-v2 not in local HF cache")
    else:
        def t_real_model():
            from transformers import AutoModel, AutoTokenizer

            gen = object.__new__(ReferenceAttackGenerator)
            gen.mode = "cos"
            gen.lambda_ppl = 0.0
            gen.suffix_prefix = "Neglect: "
            gen.checkpoint_targets = DEFAULT_CHECKPOINT_TARGETS
            gen.embed_tokenizer = AutoTokenizer.from_pretrained(snap)
            gen.embed_model = AutoModel.from_pretrained(snap).to(device).eval()
            gen.embedding_weight = gen.embed_model.get_input_embeddings().weight
            gen.vocab_size = gen.embedding_weight.size(0)
            gen._compute_ppl = lambda text: 50.0  # LM stub: stay offline
            out = gen.optimize_suffix(
                "what is the capital of france",
                "which city serves as the capital of france",
                suffix_len=5, steps=3, batch_size=4, top_k=8,
            )
            assert len(out["cost_trace"]) == 3 and out["suffix"] is not None
            checks.append(("real_model_smoke", True, f"final_cosine={out['final_cosine']}"))

        try:
            t_real_model()
        except Exception as e:  # noqa: BLE001
            checks.append(("real_model_smoke", False, repr(e)))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    n_ok = sum(1 for _, ok, _ in checks if ok)
    overall = n_ok == len(checks)
    print(f"SELFTEST {'PASS' if overall else 'FAIL'} ({n_ok}/{len(checks)})")
    return 0 if overall else 1


# --------------------------------------------------------------------------- CLI


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.selftest:
        return _selftest()
    if not args.tasks:
        parser.error("--tasks is required unless --selftest")

    tasks = shard_tasks(load_tasks(args.tasks), args.shard_index, args.shard_count)
    ckpts = tuple(float(x) for x in args.checkpoint_targets.split(","))
    logger.info("tasks in shard %d/%d: %d", args.shard_index, args.shard_count, len(tasks))

    generator = ReferenceAttackGenerator(
        embed_model_name=args.embed_model,
        lm_model_name=args.lm_model,
        mode="cos",
        lambda_ppl=args.lambda_ppl,
        checkpoint_targets=ckpts,
    )
    rows = run_tasks(
        tasks,
        generator,
        steps=args.steps,
        batch_size=args.batch_size,
        top_k=args.top_k,
        suffix_len=args.suffix_len,
        embed_model_name=args.embed_model,
        lambda_ppl=args.lambda_ppl,
        checkpoint_targets=ckpts,
        trace_path=args.trace_output,
        output_path=args.output,
        verbose=args.verbose,
    )
    # rows were already appended incrementally by run_tasks; a rewrite here
    # would drop everything recovered from prior (crashed) runs.
    output_file = Path(args.output)
    total_rows = (
        sum(1 for _ in output_file.open(encoding="utf-8"))
        if output_file.exists()
        else 0
    )
    summary = {
        "tasks": len(rows),
        "tasks_total_in_output": total_rows,
        "output": args.output,
        "trace_output": args.trace_output,
        "lambda_ppl": args.lambda_ppl,
        "embed_model": args.embed_model,
        "steps": args.steps,
        "mean_final_cosine": round(sum(r["final_cosine"] for r in rows) / len(rows), 4) if rows else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
