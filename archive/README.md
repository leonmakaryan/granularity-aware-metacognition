# Earlier experiments

This folder keeps the code and results of the experiments that came before the final
method in `pipeline/`. The thesis appendix "Development Experiments" describes them and
what we learned from them.

The code is kept as it ran, with shorter comments. It uses older prompts, target rules and
scoring rules than the final pipeline, so its numbers are not comparable with the thesis
results, and it is not maintained. Most runs used Qwen2.5-1.5B-Instruct and the 61 test
questions of relation P131 ("located in"). Files and adapters named `8b` are
Qwen2.5-7B-Instruct.

| Folder | What it was |
|---|---|
| `1_baselines/` | Inference baselines: the bare question, a "do you know?" question, and DRAG from the GRANOLA paper. |
| `2_grpo/` | GRPO with rewards for correctness and specificity, first with GranuScore. The best reward gave unstable results over four seeds. |
| `3_sft/` | The capability curve as a training target, light multi-task SFT, DPO and KTO, the prompt baselines, and a sheet for a human check of the place matcher. DPO and KTO overshot; SFT was stable. |
| `4_dates/` | The dates domain: base model, sample cache, a fixed-level prompt control, and the dates SFT. |
| `5_scale_7b/` | The location experiments at 7B, including the "surgical" target (keep the base model's correct answers, abstain where it was confidently wrong). |
| `6_transfer_joint/` | Adapters trained on one domain and evaluated on the other, and joint training on both. |
| `7_probes/` | The first runs of the conflicting-sources probe and of the entity relations. |
| `8_rescoring/` | Offline rescoring after two scoring fixes, bootstrap intervals, a weight ablation, and a dataset export with checksums. |
| `9_level_readout/` | A readout that picks the output level from the model's own signals (log-probabilities, a yes/no self-check, hidden states). It was not used in the thesis. These modules belonged to `pipeline/`. |
| `10_pipeline_checks/` | One-off checks made while running the final pipeline. |
| `src/` | The shared library of the earlier code. |
| `phase1_results/`, `phase2_results/` | Saved answers and summaries of these runs. LoRA adapters are not included. |

To run a script, start from this folder so that `src` and `phase2_results/` resolve:

    cd archive
    PYTHONPATH=. python 2_grpo/phase2.py

`pipeline/checks.py` uses a few files in `phase2_results/` to check that the final scorer
reproduces the numbers of the earlier code.
