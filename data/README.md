# Data

This directory is the data root for the scripts (`export SENTRY_DATA_ROOT=$PWD/data`).
It holds the benchmark entries, benign controls, cached answers with outcome labels, and
the per-row scores behind the paper. `MANIFEST.json` lists every file with its size,
SHA256 and, for JSONL files, its row count.

File names use internal set names: `lmp` = CAP, `scp` = SCP, `kca` = KCA.

## Start here

`runs/supp_appendix_20260923/inputs/rows_{lmp,scp,kca}.jsonl` contain one row per cache
entry of the main evaluation (attack entries and same-corpus benign controls):

| Field | Meaning |
|---|---|
| `arm` | `attack` or `genuine` (benign control) |
| `family` | construction (e.g. `ndss_matched_blend` for CAP blend, `human_comqa` for benign) |
| `key`, `query`, `answer` | cache key, incoming query, cached answer |
| `poisoned` | outcome-judge label of the cached answer |
| `base_cos` | cosine similarity between key and query (e5-small-v2, CLS pooling) |
| `excess_span`, `adl_best`, `echo_best` | Deletion Gain, ADL and Echo under the 4+2 segmentation |
| `intent_id` | intent group used for splits and bootstrap |

`python -m experiments.paper.verify_main_table` recomputes Table 1 for our defense
from these three files.

## Layout

| Path | Contents |
|---|---|
| `datasets/final500/sets/` | The three 800-entry selections (CAP, SCP, KCA) with their sampling rule and seed |
| `datasets/final500/eval/` | Evaluation records: attack entries, benign controls and incoming queries; `poisoned_flags.jsonl` holds the outcome labels |
| `datasets/answer_check_20260909/sets/` | Manifests of the harmless-instruction sets |
| `datasets/qqp/` | QQP cohorts for the cross-dataset transfer |
| `responses/answer_check_20260909/answers/` | Qwen3-8B backend answers for attack entries and harmless-instruction keys |
| `responses/answer_check_20260909/b2/` | Judge inputs and verdicts of the known-query gradient attack |
| `runs/paper/perrow/` | Per-row scores of Deletion Gain and every baseline, bootstrap intervals and budgets |
| `runs/paper/granularity32/`, `runs/paper/figures/` | Segmentation sweep and frozen figure inputs |
| `runs/paper/operating/` | Baseline score tables and cross-judge agreement samples |
| `runs/answer_check_20260909/` | Answer Check cells for four embedding models, ASR rows and adaptive-attack rows |
| `runs/supp_appendix_20260923/` | Per-row tables with the labelled cached answers, the judge runs of the answer-aware baselines, and the QQP transfer rows |
| `runs/sc_ipi_ablation_20260920/` | Rows of the Answer Check ablation on KCA |
| `runs/serving_20260916/`, `runs/asr_columns_20260916/` | Latency samples and LaCache scores |

## Sources and licenses

| Source | Use | License |
|---|---|---|
| ComQA (Abujabal et al., 2019) | CAP and SCP target questions, benign controls | released by its authors for research |
| Natural Questions (Kwiatkowski et al., 2019) | KCA target questions, benign controls | CC BY-SA 3.0 |
| Quora Question Pairs (Iyer et al., 2017) | Cross-dataset transfer | Quora terms of use (non-commercial) |

Rows derived from a dataset keep that dataset's license. Generated entries, answers and
labels are released for research use. CAP entries were generated with DeepSeek-v4-flash,
backend answers with Qwen3-8B, and outcome labels with DeepSeek-v4-flash.

## Content warning

The attack entries contain deliberately incorrect answers and injected instructions, and
some cached answers contain harmful text produced by the backend model. Use them for
defensive research only.
