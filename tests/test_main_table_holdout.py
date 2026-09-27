import numpy as np

from experiments.paper.rq1_detection.main_table_holdout import (
    joint_decisions, rates, score_decisions, split_masks,
)
from experiments.paper.rq1_detection.v3_detect import DumpedRow


def test_shared_intents_never_cross_calibration_and_test_arms():
    benign = [f"intent-{i}" for i in range(100)] * 2
    attack = [f"intent-{i}" for i in range(150)] * 3
    for seed in range(10):
        fit_b, test_b, test_a = split_masks(benign, attack, seed)
        fit_ids = set(np.asarray(benign)[fit_b])
        test_ids = set(np.asarray(benign)[test_b]) | set(np.asarray(attack)[test_a])
        assert fit_ids.isdisjoint(test_ids)
        assert fit_b.any() and test_b.any() and test_a.any()
        for intent in set(benign) & set(attack):
            assert set(test_b[np.array(benign) == intent]) == set(test_a[np.array(attack) == intent])


def test_score_threshold_ignores_test_scores_and_keeps_ties_unblocked():
    benign = np.array([0.0] * 95 + [1.0] * 5 + [100.0])
    fit = np.array([True] * 100 + [False])
    attack = np.array([0.0, 1.0])
    _, before = score_decisions(benign, attack, fit)
    benign[-1] = -100.0
    _, after = score_decisions(benign, attack, fit)
    assert before.tolist() == after.tolist() == [False, True]
    tied = np.array([0.0] * 90 + [1.0] * 10)
    blocked_b, blocked_a = score_decisions(tied, attack, np.ones(100, bool))
    assert not blocked_b.any() and not blocked_a.any()


def test_rates_use_only_test_denominators_and_frozen_success_subset():
    actual = rates(np.array([True, False, True]), np.array([True, False, True, False]),
                   np.array([False, True, True]), np.array([False, True, True, True]),
                   np.array([True, False, True, False]))
    assert actual == {"fpr": 0.5, "br_all": 1 / 3, "br_success": 1.0}


def test_joint_calibration_excludes_benign_test_answer_evidence():
    def row(i, gain, loss):
        return DumpedRow("genuine", f"i-{i}", f"r-{i}", "test", 0.96, 12,
                         gain, loss, 1, None, True)
    benign = [row(i, i / 100, i / 100) for i in range(20)]
    attack = [row(30, 0.5, 0.5), row(31, 0.0, 0.0)]
    fit = np.array([True] * 10 + [False] * 10)
    _, before = joint_decisions(benign, attack, fit)
    benign[10:] = [row(i, 100.0, 100.0) for i in range(10, 20)]
    _, after = joint_decisions(benign, attack, fit)
    assert before.tolist() == after.tolist() == [True, False]
