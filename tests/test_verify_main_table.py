"""The released per-row tables reproduce the Table 1 numbers of our defense."""
from pathlib import Path

import pytest

from experiments.paper.verify_main_table import CLASSES, ROWS, evaluate, read_rows

DATA = Path(__file__).resolve().parents[1] / 'data'
EXPECTED = {  # AUC(DG), BR, ASR, ASR without a defense
    'CAP': (0.963, 0.820, 0.006, 0.300),
    'SCP': (0.991, 0.956, 0.026, 0.802),
    'KCA': (0.992, 0.982, 0.014, 0.916),
}


@pytest.mark.skipif(not (DATA / ROWS).exists(), reason='bundled data not present')
@pytest.mark.parametrize('name,stem', CLASSES)
def test_table1_row(name, stem):
    r = evaluate(read_rows(DATA / ROWS / f'rows_{stem}.jsonl'), 0.05)
    got = (r['auc_dg'], r['br'], r['asr'], r['asr_no_defense'])
    assert [round(x, 3) for x in got] == list(EXPECTED[name])
    assert abs(r['benign_fpr'] - 0.05) < 0.005
