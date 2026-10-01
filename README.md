# Granularity-Aware Metacognition in Large Language Models

Code and saved results of my master's thesis at the Technical University of Munich
(supervisor: Georg Groh, advisor: Lukas Ellinger, October 2026).

Language models often give a specific answer when they know only part of it, for example an
exact date when they know only the year. The thesis asks if a model can learn to choose the
level of detail of its answers: as much correct detail as it can reliably give, less when it
knows less, and "I don't know" when it knows nothing.

## Method

1. **Sample.** The base model answers every training question eight times.
2. **Target.** The finest level at which at least half of the eight answers are correct
   becomes the target (for dates: day, month, year, decade or century; for places: the
   entries of the answer hierarchy). If no level reaches half, the target is "I don't know".
3. **Train.** One of the model's own answers, shortened to the target level, becomes the
   training answer. LoRA adapters are fine-tuned on these examples, for Qwen2.5-Instruct
   with 1.5B, 3B and 7B parameters, three seeds each.
4. **Evaluate.** A correct answer earns more the more detail it gives, an abstention earns
   zero, and a wrong answer loses what it would have earned if it were correct.

## Main results

- Training teaches the models to back off: on dates at 1.5B and 7B, and on locations at 3B
  and 7B.
- At 7B, the date adapters raise the score by +0.180 [+0.127, +0.235] (95% interval),
  mainly by giving years instead of wrong exact dates.
- On a separate validation set, the 7B location adapters raise it by +0.032
  [+0.005, +0.062], mainly by abstaining instead of naming a wrong place.
- Training does much less to add detail when the model knows more.

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | The method: sampling, targets, training, scoring, evaluation, reports. `pipeline/__init__.py` lists the modules. |
| `src/` | Dataset loading, prompts, the place matcher and the date scorer, used by the pipeline. |
| `analysis/` | Scripts that recompute every number, table and figure of the thesis from the saved answers. |
| `results/` | Saved samples, answers, probes and reports per model (`1p5b`, `3b`, `7b`), and the hand-checked answers of the scorer check (`scorer_check/`). |
| `scripts/run.sh` | Runs the pipeline for one model, step by step. |
| `data/` | The persons list for the date questions. |
| `third_party/` | Lukas Ellinger's temporal parser (`ugc`) and level widths (`evaluation`). |
| `archive/` | The earlier experiments (GRPO, DPO, KTO, the first SFT recipes) with their results. See `archive/README.md`. |

## Setup

Python 3.12 or newer, from the repository root:

    pip install -r requirements.txt

The location questions load from the Hugging Face dataset `lukasellinger/granola-eq`. If it
asks for access, request it on Hugging Face and put `HF_TOKEN=...` into a `.env` file in the
repository root. The models download from Hugging Face on first use.

## Recompute the thesis numbers (no GPU)

    CONFIRM=1 PYTHONPATH=. python analysis/make_tables.py

This recomputes every number of the results chapter and its appendix from `results/`,
checks each one against the stored reports, and writes the tables and figures to
`analysis/output/` and every reported number to `results/tables.json`. `CONFIRM=1` allows
reading the location validation set, which the pipeline otherwise refuses (it was used once,
at the end).

Other checks:

    PYTHONPATH=. python -m pipeline.checks                    # tests of the scoring rules
    PYTHONPATH=. python -m pipeline.analysis_tests /tmp/test  # tests of the report code
    CONFIRM=1 PYTHONPATH=. python analysis/check_consistency.py

`check_consistency.py` also rebuilds the training examples of every adapter, so it needs the
adapters in `results/<model>/adapters/`.

## Run the pipeline (GPU)

    GPU=0 MODEL=7b bash scripts/run.sh gates smoke sample
    GPU=0 MODEL=7b bash scripts/run.sh pilot select central
    GPU=0 MODEL=7b bash scripts/run.sh confirm
    GPU=0 MODEL=7b bash scripts/run.sh extras probes binary esma

`GPU` is the card's index in `nvidia-smi`. Each step leaves a marker in
`results/<model>/done/` and is skipped on a rerun. For the thesis, the 7B model went through
every phase, the 1.5B model ran `gates sample central probes binary esma`, and the 3B model
`gates sample central`; both reuse the number of epochs chosen by the 7B pilot. All settings
are in `pipeline/config.py`. The runs used an NVIDIA A40 (48 GB); the 1.5B model also fits on
a 12 GB card.

## Adapters

The LoRA adapters are too large for this repository. `results/ARTIFACTS.sha256` lists the
checksums of their files.

## Data and credits

- Location questions: GRANOLA-EQ (Yona et al., 2024), in the version
  `lukasellinger/granola-eq` (revision `afa1865`), which adds a level value to every
  hierarchy entry.
- Date questions: a list of 1666 people with their dates of birth and death from Wikidata,
  from unpublished, ongoing research by Lukas Ellinger (`data/persons_metadata.jsonl`).
- The temporal parser and the date-scoring approach in `third_party/` come from the same
  research by Lukas Ellinger.

## Citation

    @mastersthesis{makaryan2026granularity,
      author = {Leon Makaryan},
      title  = {Granularity-Aware Metacognition in Large Language Models},
      school = {Technical University of Munich},
      type   = {Master's thesis},
      year   = {2026}
    }
