"""The decision boundary for ``excess``: a quantile surface over ``(cos, log words)``.

``excess`` is a difference of cosines, and a text that already sits at 0.983 against
its anchor has at most 0.017 of room to improve by deletion, while one at 0.936 has
0.064 (``docs/METHOD.md`` §2). A single global threshold therefore reads
*distance* at one end of the band and the statistic at the other. Every headline number
in §3 is reported inside a matched cosine band for exactly this reason, so a deployed
boundary that ignored cosine would not be enforcing what those numbers measured.

    excess_max(cos, words) = a + b·(cos − c̄) + c·(log words − w̄)

fitted as the ``1 − budget`` conditional quantile. ``log words`` is the second control
§2 names: attack texts are longer than benign queries in some families.

:meth:`ExcessFence.fit_flat` keeps one height for every pair. It is the honest
comparator whenever two statistics are compared at a matched budget, and it is the form
that reproduces §4's fence-transfer result literally.

The fit is numpy-only, so the serving path carries no extra dependency. It is ported
from the deleted ``threshold_model.py`` (tag ``v1.0-paraphrase-residual``): its
centring and minimum-norm
least-squares step are not incidental, they are what stop a degenerate calibration set
from producing a boundary that extrapolates absurdly outside the range it saw.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

FEATURES = ("intercept", "cosine", "log_words")
#: Only ``a2`` ships. ``a1`` is nameable so that a fence fitted for the query side, if
#: one is ever fitted, cannot be applied to entry-side profiles by accident.
#: Which text the statistic scores. ``entry`` scores the cached entry against the
#: arriving query -- the side the threat model needs, and the only one deployed, because
#: the attacker owns the entry and the victim's query is clean. ``query`` is the reverse
#: and is measured but not served.
DIRECTIONS = ("entry", "query")

#: Fences fitted before the rename carry the old positional labels.
_LEGACY_DIRECTIONS = {"a2": "entry", "a1": "query"}


def canonical_direction(value: str) -> str:
    """Accept a legacy ``a1``/``a2`` label so already-fitted fences still load."""
    return _LEGACY_DIRECTIONS.get(str(value), str(value))
STATISTICS = ("excess_span", "excess_del")

# Grid the bounded-refit guardrail compares two boundaries over.
_DRIFT_COSINES = (0.90, 0.93, 0.96, 0.99)
_DRIFT_WORDS = (4, 8, 16, 32, 64)


@dataclass(frozen=True)
class CalibrationRow:
    """One benign observation: what the boundary is fitted to."""

    cosine: float
    words: int
    excess: float
    #: How much the entry's match to its own cached answer falls when the winning
    #: variant's complement goes. None for rows logged without an answer, and for every
    #: row written before the answer check existed; :meth:`ExcessFence.fit_flat` learns
    #: ``eta_a`` over the rows that carry one.
    answer_loss: Optional[float] = None


def _design(cosine: np.ndarray, words: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Design matrix with the two slope features centred.

    Centring is what makes a degenerate calibration set safe. If every row sits at one
    text length then ``log words`` is a *constant* column, proportional to the intercept
    column; an uncentred minimum-norm fit spreads the level across all three, handing a
    real coefficient to a feature the data says nothing about, and the boundary then
    drifts at every other length. Centred, that column is identically zero and the
    coefficient is too.
    """
    return np.column_stack([
        np.ones(len(cosine)),
        cosine - centers[0],
        np.log(words) - centers[1],
    ])


def _fit_quantile(design: np.ndarray, target: np.ndarray, quantile: float,
                  iterations: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Linear quantile regression by iteratively reweighted least squares.

    Majorise-minimise on the pinball loss: the asymmetric weight picks the quantile, the
    ``1/|residual|`` factor turns the L1 objective into a weighted L2 one.

    Each step is solved as a *minimum-norm* least-squares problem rather than by
    inverting the normal equations. Calibration traffic is frequently degenerate -- a
    narrow cosine range, or one text length -- which leaves a feature with no variation
    and the normal equations singular. Inverting anyway yields enormous coefficients
    that cancel on the training rows and then extrapolate absurdly elsewhere. The
    minimum-norm solution instead leaves an uninformative feature near zero, which is
    the least-committal reading of data that says nothing about it.
    """
    coefficients = np.zeros(design.shape[1])
    coefficients[0] = float(np.quantile(target, quantile))
    for _ in range(iterations):
        error = target - design @ coefficients
        weights = np.sqrt(
            np.where(error > 0, quantile, 1.0 - quantile) / np.maximum(np.abs(error), 1e-6))
        step, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
        if not np.all(np.isfinite(step)):
            break
        converged = float(np.max(np.abs(step - coefficients))) < tol
        coefficients = step
        if converged:
            break
    return coefficients


@dataclass
class ExcessFence:
    """``excess_max = a + b·cos + c·log(words)``; block when ``excess`` exceeds it."""

    coefficients: np.ndarray
    budget: float
    #: Means subtracted from ``(cosine, log words)`` before the slopes apply.
    centers: np.ndarray = field(default_factory=lambda: np.zeros(2))
    #: Which text this was fitted to score; see :data:`DIRECTIONS`.
    direction: str = "entry"
    statistic: str = "excess_span"
    embedder: str = "unknown"
    #: :meth:`~sentry.cache.defense.spans.SpanPolicy.fingerprint` of the cut this was
    #: fitted under.
    policy: str = "unknown"
    n_rows: int = 0
    metadata: dict = field(default_factory=dict)
    #: Which answer check a DG veto must also pass; see ``deletion.ANSWER_RULES``.
    answer_rule: str = "none"
    #: ``answer_loss`` ceiling for the ``adl`` / ``either`` rules; None otherwise.
    eta_a: Optional[float] = None
    #: minimum echo count for the ``echo`` / ``either`` rules.
    echo_min: int = 1

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=float).reshape(-1)
        if self.coefficients.shape != (len(FEATURES),):
            raise ValueError(
                f"expected {len(FEATURES)} coefficients, got {self.coefficients.shape}")
        self.centers = np.asarray(self.centers, dtype=float).reshape(-1)
        if self.centers.shape != (2,):
            raise ValueError(f"expected 2 centers, got {self.centers.shape}")
        if not 0.0 < self.budget < 1.0:
            raise ValueError("budget must lie in (0, 1)")
        self.direction = canonical_direction(self.direction)
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"unknown direction {self.direction!r}; expected one of {DIRECTIONS}")
        if self.statistic not in STATISTICS:
            raise ValueError(
                f"unknown statistic {self.statistic!r}; expected one of {STATISTICS}")
        from .deletion import ANSWER_RULES  # local import: deletion imports nothing from here
        if self.answer_rule not in ANSWER_RULES:
            raise ValueError(f"unknown answer rule {self.answer_rule!r}")
        if self.answer_rule in ("adl", "either") and self.eta_a is None:
            raise ValueError(f"answer rule {self.answer_rule!r} needs eta_a")
        if self.echo_min < 1:
            raise ValueError("echo_min must be >= 1")

    # ---- construction ----------------------------------------------------
    @classmethod
    def fit(cls, rows: Iterable[CalibrationRow], budget: float = 0.05,
            embedder: str = "unknown", policy: str = "unknown",
            direction: str = "entry", statistic: str = "excess_span",
            answer_rule: str = "none", eta_a: Optional[float] = None,
            echo_min: int = 1) -> "ExcessFence":
        """Fit the ``1 − budget`` conditional quantile on benign rows.

        Rows must be legitimate traffic only. For ``entry`` that means genuine *entries*
        scored against benign queries for their intent -- the same-corpus control that
        ``docs/METHOD.md`` §5 shows is not optional.

        This is a real precondition and nothing here can check it. In particular the
        serving path's calibration log holds every *evaluated* candidate, attacks
        included, on purpose -- so ``InMemoryCalibrationLog.rows()`` is not an argument
        for this method. Its ``anchor_only=True`` tier is; see
        ``DeletionVetoEvaluation.refit_fence``. Fitting a benign quantile over rows that
        include attacks lifts the boundary exactly where the attacks are.
        """
        usable = cls._usable(rows)
        if len(usable) < len(FEATURES) + 1:
            raise ValueError(
                f"need at least {len(FEATURES) + 1} calibration rows, got {len(usable)}")
        cosine, words, excess = cls._columns(usable)
        centers = np.array([float(cosine.mean()), float(np.log(words).mean())])
        coefficients = _fit_quantile(
            _design(cosine, words, centers), excess, 1.0 - budget)
        return cls(coefficients, budget, centers=centers, direction=direction,
                   statistic=statistic, embedder=embedder, policy=policy,
                   n_rows=len(usable), metadata=cls._metadata(cosine, words),
                   answer_rule=answer_rule, eta_a=eta_a, echo_min=echo_min)

    @classmethod
    def fit_flat(cls, rows: Iterable[CalibrationRow], budget: float = 0.05,
                 embedder: str = "unknown", policy: str = "unknown",
                 direction: str = "entry", statistic: str = "excess_span",
                 answer_rule: str = "none", echo_min: int = 1) -> "ExcessFence":
        """One height for every pair: the unconditional ``1 − budget`` benign quantile.

        Same object with both slopes pinned to zero, so everything that consumes a
        boundary takes it unchanged.
        """
        usable = cls._usable(rows)
        if not usable:
            raise ValueError("need at least one calibration row")
        cosine, words, excess = cls._columns(usable)
        height = float(np.quantile(excess, 1.0 - budget))
        metadata = cls._metadata(cosine, words)
        metadata["form"] = "flat"
        # The answer ceiling is the same kind of object as the excess ceiling: the
        # ``1 - budget`` quantile of benign traffic. It is learned only for the rules
        # that read it, so an ``echo`` fence never carries a number nothing consults.
        eta_a = None
        if answer_rule in ("adl", "either"):
            losses = np.array([r.answer_loss for r in usable if r.answer_loss is not None],
                              dtype=float)
            if not losses.size:
                raise ValueError(f"answer rule {answer_rule!r} needs rows with answer_loss")
            eta_a = float(np.quantile(losses, 1.0 - budget))
        return cls(np.array([height, 0.0, 0.0]), budget,
                   centers=np.array([float(cosine.mean()), float(np.log(words).mean())]),
                   direction=direction, statistic=statistic, embedder=embedder,
                   policy=policy, n_rows=len(usable), metadata=metadata,
                   answer_rule=answer_rule, eta_a=eta_a, echo_min=echo_min)

    @staticmethod
    def _usable(rows: Iterable[CalibrationRow]) -> list[CalibrationRow]:
        return [r for r in rows if r.words >= 1 and math.isfinite(r.excess)]

    @staticmethod
    def _columns(rows: Sequence[CalibrationRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.array([r.cosine for r in rows], dtype=float),
                np.array([r.words for r in rows], dtype=float),
                np.array([r.excess for r in rows], dtype=float))

    @staticmethod
    def _metadata(cosine: np.ndarray, words: np.ndarray) -> dict:
        """What the fit actually saw.

        A boundary fitted over one cosine band carries no information about the others,
        and its near-zero slope is indistinguishable from a genuine finding of no cosine
        effect unless the span is reported alongside it.
        """
        return {
            "cosine_span": [float(cosine.min()), float(cosine.max())],
            "word_span": [int(words.min()), int(words.max())],
            "distinct_words": int(len(np.unique(words))),
        }

    @property
    def is_flat(self) -> bool:
        """Whether this is one height rather than a surface over ``(cos, words)``."""
        return bool(np.all(self.coefficients[1:] == 0.0))

    # ---- use -------------------------------------------------------------
    def predict(self, cosine: float, words: int) -> float:
        """The ``excess`` ceiling for a pair at this cosine and text length."""
        if words < 1:
            raise ValueError("words must be >= 1; a text with no words has no spans")
        a, b, c = self.coefficients
        return float(a
                     + b * (float(cosine) - self.centers[0])
                     + c * (math.log(float(words)) - self.centers[1]))

    def blocks(self, cosine: float, words: int, excess: float,
               answer_loss: Optional[float] = None,
               echo: Optional[int] = None) -> bool:
        """Whether this pair is blocked. A non-finite ``excess`` **blocks**.

        NaN compares False against everything, so the naive ``excess > predict(...)``
        would serve a pair whose reading is not a number -- under every fence, however
        strict. And a non-finite ``excess`` is not a small reading, it is a broken
        vector: some variant embedded to NaN or overflowed. The serving path reduces
        every unknown to a miss, and this is one.

        (The inverted policy is inherited from the statistic this replaced, where a
        degenerate reading meant an exact duplicate -- the safest hit there is. For
        ``excess`` the same arithmetic means the opposite thing.)

        Calibration is unaffected: :func:`achieved_block_rate` drops non-finite rows
        before it counts, and :meth:`fit` drops them before it fits, so neither reads
        this branch.
        """
        if not math.isfinite(excess):
            return True
        if excess <= self.predict(cosine, words):
            return False
        if self.answer_rule == "none" or answer_loss is None or echo is None:
            return True          # DG-only: no answer fields to rescue with
        # ``eta_a`` is None here only for a fence whose rule does not read it, or one
        # that was mutated into an impossible state after construction. The second case
        # is a rule that cannot read its own ceiling, and the fail-closed reading of
        # that is to uphold the veto rather than waive it.
        by_loss = True if self.eta_a is None else answer_loss > self.eta_a
        by_echo = echo >= self.echo_min
        return {"adl": by_loss, "echo": by_echo, "either": by_loss or by_echo}[self.answer_rule]

    def check_profile(self, profile) -> None:
        """Raise unless ``profile`` was built the way this fence was fitted.

        The failure this prevents is silent rather than loud: a fence fitted under
        ``count:6`` and applied to ``width:3`` profiles raises nothing, blocks nothing,
        and holds no false-block budget. It simply stops being a defense, and the served
        output looks identical either way.

        An *unidentified* fence -- one whose ``embedder`` or ``policy`` is still
        ``"unknown"``, which is what a bare ``ExcessFence.fit(rows)`` produces -- is
        refused for the same reason rather than waved through. It cannot be checked
        against anything, so accepting it would mean the guard silently switches itself
        off in exactly the case it was written for: an operator sweeps span policies,
        forgets ``--policy``, attaches the result, and every profile is judged by a
        boundary fitted for some other cut. The serving path turns this raise into a
        miss, so the cost of getting it wrong is a cold cache rather than a defenceless
        one.
        """
        if self.policy == "unknown" or self.embedder == "unknown":
            raise ValueError(
                f"fence is unidentified (embedder={self.embedder!r}, "
                f"policy={self.policy!r}) and cannot be matched against a profile; fit "
                f"it with the embedder name and the span-policy fingerprint it was "
                f"calibrated under")
        if profile.policy != self.policy:
            raise ValueError(
                f"fence was fitted under span policy {self.policy!r} but the profile "
                f"was cut under {profile.policy!r}; refit the fence under the served "
                f"policy rather than reusing one calibrated for the other")
        if profile.embedder != self.embedder:
            raise ValueError(
                f"fence was fitted on embedder {self.embedder!r} but the profile was "
                f"built with {profile.embedder!r}; a boundary is only meaningful in the "
                f"space it was measured in")

    # ---- refitting under a drift bound -----------------------------------
    def bounded_refit(self, rows: Iterable[CalibrationRow], max_drift: float = 0.05,
                      ) -> tuple["ExcessFence", Optional[str]]:
        """Refit, but only adopt the result if the boundary barely moved.

        Returns ``(fence, rejection)``: on rejection the *current* fence is returned
        unchanged along with a reason, so a caller that ignores the second element keeps
        serving the old boundary.

        This bounds two hazards with one check. Refitting on served traffic alone
        censors the upper tail and ratchets the threshold down every epoch until it
        blocks everything; and an attacker who floods traffic could otherwise walk the
        boundary loose. Callers must additionally log every *evaluated* candidate, not
        only served ones -- see the plugin.

        The refit keeps this fence's form. A flat boundary that came back sloped from
        its first epoch would silently change what is being served, and the drift check
        would not catch it, because two surfaces can agree on the reference grid while
        differing where it matters.
        """
        fit = ExcessFence.fit_flat if self.is_flat else ExcessFence.fit
        # ``fit_flat`` relearns ``eta_a`` from the new rows' losses; ``fit`` has no
        # quantile step to relearn it with, so the current ceiling is carried across --
        # without it a conditional fence under ``adl``/``either`` cannot be refitted at
        # all, because the rebuilt object would be missing the number its rule reads.
        carried = {} if self.is_flat else {"eta_a": self.eta_a}
        candidate = fit(rows, budget=self.budget, embedder=self.embedder,
                        policy=self.policy, direction=self.direction,
                        statistic=self.statistic, answer_rule=self.answer_rule,
                        echo_min=self.echo_min, **carried)
        drift = self.max_divergence(candidate)
        if drift > max_drift:
            return self, f"drift {drift:.4f} exceeds bound {max_drift:.4f}"
        return candidate, None

    def max_divergence(self, other: "ExcessFence") -> float:
        """Largest boundary gap between two fences over a reference grid."""
        return max(
            abs(self.predict(c, w) - other.predict(c, w))
            for c in _DRIFT_COSINES
            for w in _DRIFT_WORDS
        )

    # ---- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "features": list(FEATURES),
            "coefficients": [float(v) for v in self.coefficients],
            "centers": [float(v) for v in self.centers],
            "budget": self.budget,
            "direction": self.direction,
            "statistic": self.statistic,
            "embedder": self.embedder,
            "policy": self.policy,
            "n_rows": self.n_rows,
            "metadata": dict(self.metadata),
            "answer_rule": self.answer_rule,
            "eta_a": None if self.eta_a is None else float(self.eta_a),
            "echo_min": self.echo_min,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ExcessFence":
        features = tuple(payload.get("features", FEATURES))
        if features != FEATURES:
            raise ValueError(f"fence features {features} do not match {FEATURES}")
        return cls(
            np.asarray(payload["coefficients"], dtype=float),
            float(payload["budget"]),
            centers=np.asarray(payload.get("centers", [0.0, 0.0]), dtype=float),
            direction=canonical_direction(payload.get("direction", "entry")),
            statistic=str(payload.get("statistic", "excess_span")),
            embedder=str(payload.get("embedder", "unknown")),
            policy=str(payload.get("policy", "unknown")),
            n_rows=int(payload.get("n_rows", 0)),
            metadata=dict(payload.get("metadata") or {}),
            answer_rule=str(payload.get("answer_rule", "none")),
            eta_a=(None if payload.get("eta_a") is None
                   else float(payload["eta_a"])),
            echo_min=int(payload.get("echo_min", 1)),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ExcessFence":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def achieved_block_rate(fence: ExcessFence,
                        rows: Sequence[CalibrationRow]) -> float:
    """Fraction of ``rows`` the fence blocks -- the calibration check that matters.

    Read against the nominal budget on held-out data. Well above it means over-blocking
    legitimate traffic; near zero means the detector has switched off and is spending
    none of its budget on attacks.

    Rows with a non-finite ``excess`` are dropped rather than counted. Serving blocks
    them (:meth:`ExcessFence.blocks`), but they are broken readings rather than
    false blocks, and counting them here would charge a measurement fault to the
    boundary's budget.
    """
    usable = [r for r in rows if r.words >= 1 and math.isfinite(r.excess)]
    if not usable:
        return float("nan")
    return float(np.mean([fence.blocks(r.cosine, r.words, r.excess) for r in usable]))
