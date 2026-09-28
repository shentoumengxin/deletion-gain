# Reproducing the paper

Every table and figure maps to a frozen result file and to the script that produced it.
Paths below are relative to the repository root. `R/` stands for
`experiments/paper/results/`, `X/` for `experiments/paper/`, and `data/` is the
bundled data root (`export SENTRY_DATA_ROOT=$PWD/data`).

Steps that embed text, generate answers or call a judge need a GPU or an API key. Every
step after that reads the released per-row files and runs on a CPU in seconds to minutes.
The quickest end-to-end check is:

```sh
export SENTRY_DATA_ROOT=$PWD/data
python -m experiments.paper.verify_main_table
```

Naming: in file names, `lmp` is CAP, `scp`/`ndss` is SCP and `kca`/`gcg` is KCA. The
deployed segmentation `multi[count:4+width:2:cap16]/runs` is written 4+2 in the paper.
Fences are fitted in the flat form at the joint height (`--fence-form flat`, `*_joint_eta`
fields).

## Main text

| Paper item | Result file(s) | Script(s) |
|---|---|---|
| Table 1, Ours and no-defense ASR | `data/runs/supp_appendix_20260923/inputs/rows_{lmp,scp,kca}.jsonl`, `R/supp_appendix_20260923/answer_alignment.json` (CAP, SCP), `R/kca_main_alignment_20260920/summary.json` (KCA) | `X/verify_main_table.py`, `X/rq1_detection/v3_detect.py`, `X/rq1_detection/answer_alignment.py`, `X/rq1_detection/kca_answer_alignment.py` |
| Table 1, baselines (BR) | `data/runs/paper/perrow/ci_table1.json` and the per-row scores beside it (`multiview_*`, `salting_*`, `eac_*`, `judge_vote_*`, `*_lacache_k20_*`) | `X/rq1_detection/ci_table1.py`, `X/baselines/` (`multiview.py`, `salting_text.py`, `eac_original.py`, `judge_vote.py`, `rq2_lacache.py`, `rq2_baselines.py`), `X/gpu/lacache_prefix.py` |
| Table 1, baseline ASR | `data/runs/asr_columns_20260916/lacache_scores.json` and the per-row files above | `X/tables/recompute_baseline_asr.py`, `X/tables/replay_lacache_scores.py` |
| Table 1, Worst BR and Worst ASR | lowest BR and highest ASR of each row over the three classes | `X/verify_main_table.py` (Ours) |
| Table 1, ms/hit | `R/answer_check_20260916/serving_microbenchmark.json`, `data/runs/paper/perrow/eac_cap_timing.json`, `data/runs/serving_20260916/` | `X/rq4_system/bench_answer_check_ms.py` |
| Table 2a, Append / Interleave / Repeat | `R/answer_check_20260909/rq4_placement_asr_42_joint.json`, no-defense column in `R/appendix_20260916/placement_cached_outcomes_audit.json` | `X/rq2_robustness/rq4_placement_asr.py` |
| Table 2a, Search | `R/supp_appendix_20260923/search_4p2.json` | `X/rq2_robustness/rq6_deletion_aware_attacker.py`, `X/rq2_robustness/supp_search_replay.py`, `X/rq2_robustness/supp_search_summary.py` |
| Table 2a, Gradient; Table 15 | `R/answer_check_20260909/adaptive_gradient_row_judged.json`, `adaptive_joint_judged.json`, `adaptive_undefended_judged.json` | `X/rq2_robustness/rq4_gradient_attacker.py` |
| Table 2b | `data/runs/paper/table4.json`, `data/runs/paper/perrow/deletion_v3_cap_{e5,bge,gte,minilm}_42.jsonl`, BR in `R/supp_appendix_20260923/answer_alignment_encoders.json` | `X/rq1_detection/table4.py` |
| Figure 3 | left: `R/supp_appendix_20260923/search_figure3_rounds.json`; right: Table 1 | `X/figures/fig6_panels.py`, `X/rq2_robustness/supp_search_figure3.py` |
| Figure 4 | `data/runs/paper/perrow/deletion_v3_cap_e5_42.jsonl` | `X/figures/fig3_coexistence.py`, `X/rq1_detection/dump_perrow_gain.py` |
| Figure 5 | `data/runs/paper/granularity32/`, `R/v3/granularity32/` | `X/figures/fig7_tradeoff_ranked.py`, `X/rq3_mechanism/rq3_granularity.py` |
| Section 5.4 (information bottleneck) | `R/supp_appendix_20260923/validity_classifier.json` | `X/rq1_detection/validity_classifier.py` |
| Section 5.4 (ablations) | `R/supp_appendix_20260923/statistic_ablations.json` | `X/rq3_mechanism/statistic_ablations.py` |
| Section 5.4 (QQP transfer) | `R/supp_appendix_20260923/qqp_transfer.json`, `data/datasets/qqp/` | `X/rq3_mechanism/qqp_transfer.py`, `X/rq3_mechanism/qqp_prompts.py` |
| Section 5.5, Table 3 (real prompts) | `R/supp_appendix_20260923/vcache_replay.json`, `vcache_local_groups.json` | `X/rq4_system/vcache_replay.py`, `X/rq4_system/vcache_local_groups.py` |

Figures 1 and 2 are illustrations. To render Figures 3 to 5 into a directory:

```sh
export SENTRY_DATA_ROOT=$PWD/data SENTRY_PAPER_DIR=/tmp/paper && mkdir -p /tmp/paper/figures
python -m experiments.paper.figures.fig6_panels
python -m experiments.paper.figures.fig3_coexistence
python -m experiments.paper.figures.fig7_tradeoff_ranked
```

## Appendix

| Paper item | Result file(s) | Script(s) |
|---|---|---|
| Table 4 (by construction) | `R/supp_appendix_20260923/answer_alignment.json`, `R/kca_main_alignment_20260920/summary.json` | as Table 1 |
| Table 5 (baseline variants, NLI) | `data/runs/paper/perrow/ci_table1.json`, `R/v3/detection/lacache_k20.json` | `X/rq1_detection/ci_table1.py`, `X/tables/recompute_nli_appendix.py` |
| Table 6 (ablations, answer-aware baselines) | `R/supp_appendix_20260923/dg_necessity.json`, `response_perplexity.json`, `judge_answer_aware.json` | `X/rq1_detection/dg_necessity.py`, `X/baselines/response_perplexity.py`, `X/baselines/judge_answer_aware_api.py`, `X/baselines/judge_answer_aware_vote.py`, `X/gpu/judge_answer_aware_score.py` |
| Table 7 (FPR budgets) | `data/runs/paper/perrow/budgets.json`, `R/supp_appendix_20260923/dg_necessity.json` | `X/rq1_detection/budgets.py` |
| Table 8 (AUC) | `data/runs/paper/perrow/ci_table1.json` (`methods.*.auc`), `data/runs/paper/perrow/*_lacache_k20_*.json` | `X/rq1_detection/ci_table1.py`, `X/baselines/rq2_lacache.py` |
| Table 9 (confidence intervals) | `data/runs/paper/perrow/ci_table1.json`, `data/runs/paper/perrow/ci_answer_check.json` | `X/rq1_detection/ci_table1.py`, `X/rq1_detection/ci_answer_check.py`, `X/rq1_detection/boot.py` |
| Table 10 (pooling) | `data/runs/paper/perrow/pool_cap_*.json`, `data/runs/paper/perrow/v3_*mean_*.json`, `R/appendix_20260916/gte_collision_recheck.json` | `X/rq1_detection/pooling_check.py` |
| Table 11 (information bottleneck) | `R/supp_appendix_20260923/validity_classifier.json` | `X/rq1_detection/validity_classifier.py` |
| Table 12 (recovery with known rewrites) | `R/supp_appendix_20260923/rewrite_recovery.json` | `X/rq3_mechanism/rewrite_recovery.py` |
| Table 13 (harmless instructions) | `R/supp_controls_20260920/frozen_replay.json`, `data/datasets/answer_check_20260909/` | `R/supp_controls_20260920/reproduce.py`, `sentry/research/pipeline/instruction_benign.py` |
| Table 14 (Answer Check ablation) | `R/sc_ipi_20260920/aligned_summary.json`, `data/runs/sc_ipi_ablation_20260920/` | `X/rq2_robustness/sc_ipi_replay.py`, `X/rq2_robustness/sc_ipi_score.py` |
| Tables 16 and 17 (GPTCache deployment and cost) | `R/supp_appendix_20260923/gptcache_workload_full_rule.json`, `R/answer_check_20260916/serving_microbenchmark.json`, `R/v3/granularity32/g32_{cap,scp,kca}.json` | `X/rq4_system/gptcache_workload_full_rule.py`, `X/rq4_system/bench_answer_check_ms.py` |
| Segmentation selection | `R/supp_appendix_20260923/sweep_full_rule.json` | `X/rq3_mechanism/sweep_full_rule.py` |
| Threshold form | `R/supp_appendix_20260923/threshold_form.json` | `X/rq3_mechanism/threshold_form.py` |
| Held-out FPR | `R/main_table_holdout_20260913/summary.json` | `X/rq1_detection/main_table_holdout.py` |
| Judge agreement (κ) | `R/v3/asr/cross_judge_v3.summary.json`, `R/v3/operating/cross_judge_phi4.summary.json` | `X/rq2_robustness/cross_judge_local.py` |
| Worked examples | `R/case_studies_20260916/summary.json` | scores from `sentry/cache/defense/deletion.py` |
| CAP benchmark generation | `data/datasets/final500/` | `sentry/research/pipeline/generate.py`, `X/rq1_detection/fix_sets.py` |
| SCP construction | `R/v3/scp_build.json`, `R/v3/scp_payloads_stats.json` | `X/rq2_robustness/scp_payloads.py`, `X/rq2_robustness/scp_generate.py` |
| KCA generation | `R/v3/kca_anchors.json` | `X/attack/reference_attack.py` (requires CacheAttack) |
| Backend answers and outcome labels | `data/responses/answer_check_20260909/`, `data/runs/supp_appendix_20260923/inputs/judged_*.jsonl` | `X/rq2_robustness/asr_generate.py`, `X/rq2_robustness/asr_judge.py`, `X/rq2_robustness/asr_defended.py --fence-form flat` |

The real-prompt replay uses the public vCache benchmarks, which are not redistributed
here. Download them from the vCache release and pass their location to the vCache scripts.
