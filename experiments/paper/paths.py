"""Paths for paper experiments; immutable evidence is external to the code tree."""
import os
from pathlib import Path
from sentry.artifacts import data_root

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / 'results'

def paper_data_root():
    return data_root() / 'runs' / 'paper'

def perrow_root():
    return paper_data_root() / 'perrow'

def figure_output_root():
    value = os.environ.get('SENTRY_PAPER_DIR')
    if not value:
        raise ValueError('Set SENTRY_PAPER_DIR or use figures.export --paper-dir.')
    return Path(value).expanduser().resolve() / 'figures'
