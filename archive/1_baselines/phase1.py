import json
import re
import string
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- config ----------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "lukasellinger/granola-eq"
SPLIT = "test"          
RELATION = "P131"
GRANOLA_NUM_SAMPLES = 10
GRANOLA_TEMPERATURE = 0.7
BASE_BATCH_SIZE = 16    
OUT_DIR = Path("phase1_results")
OUT_DIR.mkdir(exist_ok=True)


# ---------- data ----------

@dataclass
class Question:
    qid: int
    question: str
    entity: str
    hierarchy: list[str]      # most specific -> coarsest, e.g. ["Loir-et-Cher", "Centre-Val de Loire", "France"]
    levels: list[float]       # parallel to hierarchy, Lukas's normalized levels (e.g. [1, 2.5, 4])

    @property
    def gold_specific(self) -> str:
        return self.hierarchy[0]


def load_questions() -> list[Question]:
    ds = load_dataset(DATASET, split=SPLIT)
    qs = []
    for row in ds:
        if row["relation"] != RELATION:
            continue
        # Sanity: gold answer should be the most specific entry.
        # We don't hard-fail because there could be edge cases; just log.
        hierarchy = list(row["granola_answers"])
        levels = list(row["normalized_levels"])
        assert len(hierarchy) == len(levels), f"length mismatch on id {row['id']}"
        if hierarchy[0].strip().lower() != row["answer"].strip().lower():
            # not fatal; some rows might have minor formatting differences
            pass
        qs.append(Question(
            qid=row["id"],
            question=row["question"],
            entity=row["question_entity"],
            hierarchy=hierarchy,
            levels=levels,
        ))
    print(f"loaded {len(qs)} {RELATION} questions from {DATASET}/{SPLIT}")
    return qs


# ---------- granularity classifier ----------

_ARTICLES = {"the", "a", "an"}

def normalize(s: str) -> str:
    s = s.lower().strip()
    # strip punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # strip leading articles
    tokens = s.split()
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def _content_tokens(s: str) -> set[str]:
    """Tokens after normalization, dropping common filler words.
    Used as a fallback match: 'united states' vs 'united states of america'."""
    filler = {"of", "the", "a", "an", "and"}
    return {t for t in normalize(s).split() if t and t not in filler}


def classify_answer(answer: str, q: Question) -> dict:
    """Match an answer against the hierarchy: exact match first, then containment,
        then all content words of an entry. Returns the matched level (0 = most specific)
        and its normalized value, or None."""
    if not answer or not answer.strip():
        return {"level_idx": None, "level_value": None, "matched_to": None, "raw": answer}

    norm_ans = normalize(answer)
    if not norm_ans:
        return {"level_idx": None, "level_value": None, "matched_to": None, "raw": answer}

    ans_tokens = _content_tokens(answer)

    # Tier 1: exact match, most specific first
    for idx, h in enumerate(q.hierarchy):
        if normalize(h) == norm_ans:
            return {"level_idx": idx, "level_value": q.levels[idx], "matched_to": h, "raw": answer}

    # Tier 2: substring match, most specific first
    for idx, h in enumerate(q.hierarchy):
        nh = normalize(h)
        if nh and (nh in norm_ans or norm_ans in nh):
            return {"level_idx": idx, "level_value": q.levels[idx], "matched_to": h, "raw": answer}

    # Tier 3: content-word subset, most specific first.
    # The hierarchy entry's content words must all appear in the answer's content words.
    # This catches "United States" matching "United States of America".
    for idx, h in enumerate(q.hierarchy):
        h_tokens = _content_tokens(h)
        if h_tokens and h_tokens.issubset(ans_tokens):
            return {"level_idx": idx, "level_value": q.levels[idx], "matched_to": h, "raw": answer}

    return {"level_idx": None, "level_value": None, "matched_to": None, "raw": answer}


# ---------- metacognition metrics (Park et al. 2026) ----------

from statistics import NormalDist

def _phi_inv(p: float) -> float:
    """Inverse standard normal CDF, clipped to avoid ±inf at extremes."""
    eps = 1e-3
    return NormalDist().inv_cdf(max(eps, min(1 - eps, p)))


def metacognition_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    """d' type 2 (Park et al. 2026), Yes ratio, YFR and NFR from (said_yes, correct) pairs."""
    if not pairs:
        return {"d_prime": None, "yes_ratio": None, "yfr": None, "nfr": None}
    n = len(pairs)
    yes = [p for p in pairs if p[0]]
    no = [p for p in pairs if not p[0]]
    correct = [p for p in pairs if p[1]]
    incorrect = [p for p in pairs if not p[1]]

    yes_ratio = len(yes) / n
    if correct and incorrect:
        hit = sum(1 for p in correct if p[0]) / len(correct)
        fa = sum(1 for p in incorrect if p[0]) / len(incorrect)
        d_prime = _phi_inv(hit) - _phi_inv(fa)
    else:
        d_prime = None
    yfr = sum(1 for p in yes if not p[1]) / len(yes) if yes else None
    nfr = sum(1 for p in no if p[1]) / len(no) if no else None
    return {
        "d_prime": round(d_prime, 3) if d_prime is not None else None,
        "yes_ratio": round(yes_ratio, 3),
        "yfr": round(yfr, 3) if yfr is not None else None,
        "nfr": round(nfr, 3) if nfr is not None else None,
    }


# ---------- model wrapper ----------

class Generator:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _chat(self, user_msg: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], max_new_tokens: int = 32, temperature: float = 0.0) -> list[str]:
        """Greedy if temperature==0, else sampled. Returns just the generated continuation."""
        formatted = [self._chat(p) for p in prompts]
        inputs = self.tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        out = self.model.generate(**inputs, **gen_kwargs)
        # strip prompt tokens
        gen_only = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    @torch.no_grad()
    def sample_n(self, prompt: str, n: int, max_new_tokens: int = 32, temperature: float = 0.7) -> list[str]:
        """N independent samples for the same prompt. Used for GRANOLA."""
        formatted = self._chat(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_only = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        return [d.strip() for d in decoded]


# ---------- methods ----------

# Vanilla baseline prompt — Yona et al. 2024, Table 7 (verbatim).
BASE_PROMPT = "Question: {q}\nAnswer:"

# Meta-question prompt — Park et al. 2026, Figure 11 (verbatim).
# Used to elicit verbalized confidence as a Yes/No self-report.
META_QUESTION_PROMPT = (
    'Do you know the answer to the following question? '
    'If you know and are sure about the answer, just return "Yes". '
    'If you don\'t know the answer or are uncertain, just return "No".\n'
    "Question: {q}"
)

# DRAG aggregation prompt — Yona et al. 2024, Table 8, "Response Aggregation" (verbatim).
# Note: this is few-shot with two worked examples. The model is instructed to output the
# most specific answer CONSISTENT WITH ALL responses, or IDK if no meaningful overlap.
DRAG_AGGREGATION_PROMPT = """You will be given a list of responses; replace them with the most specific answer that is still consistent with all the original responses. If the responses have nothing meaningful in common with respect to the question, output IDK.

Here are some examples:

Question: Where was [X] born?
Responses:
- Hamburg
- Hamburg
- Bonn
- Berlin
Correct aggregated answer: Germany
Incorrect aggregated answer: Hamburg
Explanation: These are all different cities in Germany. Hamburg is not a correct aggregation, since it is not consistent with other responses, such as Berlin or Bonn.

Question: When was [X] born?
Responses:
- February 1, 1937
- November 20, 1937
- January 1937
Correct aggregated answer: 1937
Incorrect aggregated answer: November 1937
Explanation: These are all dates in 1937.

Question: {q}
Responses:
{candidates}
Correct aggregated answer:"""


def run_base_direct(gen: Generator, questions: list[Question]) -> list[dict]:
    results = []
    prompts = [BASE_PROMPT.format(q=q.question) for q in questions]

    for i in range(0, len(questions), BASE_BATCH_SIZE):
        batch_q = questions[i:i + BASE_BATCH_SIZE]
        batch_p = prompts[i:i + BASE_BATCH_SIZE]
        outs = gen.generate_batch(batch_p, max_new_tokens=32, temperature=0.0)
        # Truncate at first newline — small models often bleed prose/reasoning after the answer.
        outs = [o.split("\n")[0].strip() for o in outs]
        for q, raw_ans in zip(batch_q, outs):
            cls = classify_answer(raw_ans, q)
            results.append({
                "qid": q.qid,
                "question": q.question,
                "gold_specific": q.gold_specific,
                "hierarchy": q.hierarchy,
                "raw_answer": raw_ans,
                "matched_to": cls["matched_to"],
                "level_idx": cls["level_idx"],
                "level_value": cls["level_value"],
            })
        if (i // BASE_BATCH_SIZE) % 5 == 0:
            print(f"  base: {i + len(batch_q)}/{len(questions)}")
    return results


def run_granola(gen: Generator, questions: list[Question]) -> list[dict]:
    """DRAG (Decoding with Response Aggregation) — Yona et al. 2024."""
    results = []
    for i, q in enumerate(questions):
        # Step 1: sample N responses (DRAG sampling stage)
        prompt = BASE_PROMPT.format(q=q.question)
        samples = gen.sample_n(prompt, n=GRANOLA_NUM_SAMPLES, max_new_tokens=32, temperature=GRANOLA_TEMPERATURE)

        # Step 2: aggregate using few-shot prompt (DRAG aggregation stage)
        candidates_str = "\n".join(f"- {s}" for s in samples)
        agg_prompt = DRAG_AGGREGATION_PROMPT.format(q=q.question, candidates=candidates_str)
        aggregated = gen.generate_batch([agg_prompt], max_new_tokens=32, temperature=0.0)[0]
        # Truncate at first newline (model often appends "Explanation: ..." after the answer).
        aggregated = aggregated.split("\n")[0].strip()

        # Detect IDK (paper allows it as a valid output)
        is_idk = aggregated.strip().upper().startswith("IDK")

        # uncertainty signal: fraction of unique normalized samples
        norm_samples = [normalize(s) for s in samples if s.strip()]
        unique_count = len(set(norm_samples))
        uncertainty = unique_count / max(len(norm_samples), 1)

        cls = classify_answer(aggregated, q) if not is_idk else {
            "level_idx": None, "level_value": None, "matched_to": None, "raw": aggregated
        }
        results.append({
            "qid": q.qid,
            "question": q.question,
            "gold_specific": q.gold_specific,
            "hierarchy": q.hierarchy,
            "samples": samples,
            "raw_answer": aggregated,
            "is_idk": is_idk,
            "matched_to": cls["matched_to"],
            "level_idx": cls["level_idx"],
            "level_value": cls["level_value"],
            "n_unique_samples": unique_count,
            "uncertainty": uncertainty,
        })
        if i % 25 == 0:
            print(f"  drag: {i + 1}/{len(questions)}")
    return results


def run_meta_question(gen: Generator, questions: list[Question], base_results: list[dict]) -> list[dict]:
    """Verbalized-confidence baseline (Park et al. 2026): ask "do you know X?" and
        commit to the direct answer only on Yes."""
    base_by_qid = {r["qid"]: r for r in base_results}
    results = []
    for i in range(0, len(questions), BASE_BATCH_SIZE):
        batch_q = questions[i:i + BASE_BATCH_SIZE]
        meta_prompts = [META_QUESTION_PROMPT.format(q=q.question) for q in batch_q]
        meta_outs = gen.generate_batch(meta_prompts, max_new_tokens=8, temperature=0.0)
        meta_outs = [o.split("\n")[0].strip() for o in meta_outs]
        for q, meta_raw in zip(batch_q, meta_outs):
            said_yes = meta_raw.lower().lstrip().startswith("yes")
            base = base_by_qid[q.qid]
            # underlying_correct = whether the direct answer would have matched, regardless of meta.
            # This is what d' / YFR / NFR need.
            underlying_correct = base["level_idx"] is not None
            if said_yes:
                results.append({
                    "qid": q.qid, "question": q.question, "gold_specific": q.gold_specific,
                    "hierarchy": q.hierarchy,
                    "raw_answer": base["raw_answer"], "meta_raw": meta_raw,
                    "said_yes": True, "is_idk": False,
                    "matched_to": base["matched_to"],
                    "level_idx": base["level_idx"], "level_value": base["level_value"],
                    "underlying_correct": underlying_correct,
                })
            else:
                results.append({
                    "qid": q.qid, "question": q.question, "gold_specific": q.gold_specific,
                    "hierarchy": q.hierarchy,
                    "raw_answer": "IDK", "meta_raw": meta_raw,
                    "said_yes": False, "is_idk": True,
                    "matched_to": None, "level_idx": None, "level_value": None,
                    "underlying_correct": underlying_correct,
                })
        if (i // BASE_BATCH_SIZE) % 5 == 0:
            print(f"  meta-q: {i + len(batch_q)}/{len(questions)}")
    return results


# ---------- metrics ----------

def summarize(name: str, results: list[dict]) -> dict:
    n = len(results)
    matched = [r for r in results if r["level_idx"] is not None]
    idk = [r for r in results if r.get("is_idk", False)]
    off = n - len(matched) - len(idk)

    acc_at_output = len(matched) / n if n else 0.0
    specific_correct = sum(1 for r in matched if r["level_idx"] == 0)
    specific_acc = specific_correct / n if n else 0.0
    avg_level = sum(r["level_value"] for r in matched) / len(matched) if matched else None

    # selective accuracy: acc on the subset that didn't say IDK (paper's "selective GRANOLA accuracy")
    non_idk = n - len(idk)
    selective_acc = len(matched) / non_idk if non_idk else 0.0

    level_dist = Counter(r["level_idx"] for r in matched)

    return {
        "method": name,
        "n": n,
        "on_hierarchy": len(matched),
        "idk": len(idk),
        "off_hierarchy": off,
        "acc_on_hierarchy": round(acc_at_output, 4),
        "selective_acc": round(selective_acc, 4),
        "specific_accuracy": round(specific_acc, 4),
        "avg_normalized_level": round(avg_level, 3) if avg_level is not None else None,
        "level_idx_distribution": dict(sorted(level_dist.items(), key=lambda x: (x[0] is None, x[0]))),
    }


# ---------- main ----------

def main():
    questions = load_questions()
    gen = Generator()

    print("\n=== running BASE DIRECT ===")
    base_results = run_base_direct(gen, questions)
    pd.DataFrame(base_results).to_csv(OUT_DIR / "base_direct.csv", index=False)
    base_summary = summarize("base_direct", base_results)
    # base direct has no meta signal — model always commits. d' is undefined here.
    base_summary["metacognition"] = {"note": "no meta signal (model always commits)"}

    print("\n=== running META-QUESTION ===")
    mq_results = run_meta_question(gen, questions, base_results)
    pd.DataFrame(mq_results).to_csv(OUT_DIR / "meta_question.csv", index=False)
    mq_summary = summarize("meta_question", mq_results)
    mq_pairs = [(r["said_yes"], r["underlying_correct"]) for r in mq_results]
    mq_summary["metacognition"] = metacognition_metrics(mq_pairs)

    print("\n=== running DRAG ===")
    granola_results = run_granola(gen, questions)
    pd.DataFrame(granola_results).to_csv(OUT_DIR / "granola.csv", index=False)
    granola_summary = summarize("drag", granola_results)
    # For DRAG, said_yes = committed. `correct` is whether the direct answer would have
    # matched, as for meta_question, so abstentions do not count as wrong and inflate d'.
    base_by_qid = {r["qid"]: r for r in base_results}
    drag_pairs = [
        (not r["is_idk"], base_by_qid[r["qid"]]["level_idx"] is not None)
        for r in granola_results
    ]
    granola_summary["metacognition"] = metacognition_metrics(drag_pairs)

    summary = {
        "base_direct": base_summary,
        "meta_question": mq_summary,
        "drag": granola_summary,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nresults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
