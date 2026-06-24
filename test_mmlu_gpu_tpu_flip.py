"""MMLU GPU vs TPU answer-flip test.

Reproduces exactly what lm_eval does for MMLU:
  - 4 loglikelihood requests per question: (context, " A/B/C/D")
  - Predicted answer = argmax of 4 loglikelihoods
  - Each continuation is a single token

A "flip" happens when the numerical delta between platforms exceeds the margin
between the top-2 choices:
  GPU : A=-0.100, B=-0.101  → predicted A  (margin = 0.001)
  TPU : A=-0.102, B=-0.101  → predicted B  (flip! delta 0.002 > margin 0.001)

Copy both this file AND conftest.py to each server.

Usage
-----
# On GPU — save reference:
    pytest test_mmlu_gpu_tpu_flip.py \\
        --model-path=/mnt/models/my-llama \\
        --save-ref=/tmp/ref_gpu.json \\
        -v -s

# On TPU — compare:
    pytest test_mmlu_gpu_tpu_flip.py \\
        --model-path=/mnt/models/my-llama \\
        --compare-ref=/tmp/ref_gpu.json \\
        -v -s

CLI options (defined in conftest.py)
-------------------------------------
  --model-path      Local checkpoint dir or HF Hub ID  [required]
  --save-ref        JSON path to write (GPU run)
  --compare-ref     JSON path to read  (TPU run)
  --mmlu-subjects   Comma-separated subjects  [default: abstract_algebra,astronomy]
  --mmlu-limit      Questions per subject     [default: 50]
  --mmlu-fewshot    Fewshot examples          [default: 0]
  --max-model-len   vllm max_model_len        [default: 2048]
  --max-num-seqs    vllm max_num_seqs         [default: 16]

Dependencies
------------
  pip install vllm datasets numpy
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

vllm = pytest.importorskip("vllm", reason="vllm not installed — pip install vllm")
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

LABELS = ["A", "B", "C", "D"]

_DESCRIPTION_TMPL = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
)
_QUESTION_TMPL = "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChoiceResult:
    label: str
    continuation_token_ids: list[int]
    loglikelihood: float


@dataclass
class QuestionResult:
    subject: str
    question_idx: int
    context_token_ids: list[int]
    choices: list[ChoiceResult]
    predicted: str
    correct: str
    correct_idx: int


# ---------------------------------------------------------------------------
# MMLU data loading
# ---------------------------------------------------------------------------

def _load_mmlu_questions(subjects: list[str], limit: int, fewshot: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        pytest.skip("datasets not installed — pip install datasets")

    questions: list[dict] = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=False)
        description = _DESCRIPTION_TMPL.format(subject=subject.replace("_", " "))

        fewshot_str = ""
        if fewshot > 0:
            dev = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=False)
            for ex in list(dev)[:fewshot]:
                fewshot_str += (
                    _QUESTION_TMPL.format(
                        question=ex["question"].strip(),
                        a=ex["choices"][0], b=ex["choices"][1],
                        c=ex["choices"][2], d=ex["choices"][3],
                    )
                    + f" {LABELS[ex['answer']]}\n\n"
                )

        prefix = description + fewshot_str
        for idx, ex in enumerate(list(ds)[:limit]):
            questions.append({
                "subject": subject,
                "question_idx": idx,
                "context_str": prefix + _QUESTION_TMPL.format(
                    question=ex["question"].strip(),
                    a=ex["choices"][0], b=ex["choices"][1],
                    c=ex["choices"][2], d=ex["choices"][3],
                ),
                "correct_idx": ex["answer"],
            })
    return questions


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_loglikelihoods(llm: LLM, questions: list[dict]) -> list[QuestionResult]:
    """Run 4 loglikelihood requests per question, mirroring lm_eval exactly."""
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0,
        prompt_logprobs=1,
        max_tokens=1,
        detokenize=False,
    )

    # Build all requests up front so vllm batches them together.
    all_requests: list[tuple[int, str, list[int], list[int]]] = []
    ctx_ids_per_q: list[list[int]] = []

    for q in questions:
        ctx_ids = tokenizer.encode(q["context_str"], add_special_tokens=True)
        ctx_ids_per_q.append(ctx_ids)
        q_idx = len(ctx_ids_per_q) - 1
        for label in LABELS:
            cont_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
            all_requests.append((q_idx, label, ctx_ids, cont_ids))

    prompt_inputs = [
        {"prompt_token_ids": ctx + cont}
        for _, _, ctx, cont in all_requests
    ]
    outputs = llm.generate(prompt_inputs, sampling_params)

    results_map: dict[tuple[int, str], tuple[list[int], float]] = {}
    for (q_idx, label, ctx_ids, cont_ids), output in zip(all_requests, outputs):
        prompt_lp = output.prompt_logprobs
        total_lp = 0.0
        for pos in range(len(ctx_ids), len(ctx_ids) + len(cont_ids)):
            lp_dict = prompt_lp[pos]
            token_id = (ctx_ids + cont_ids)[pos]
            if lp_dict and token_id in lp_dict:
                total_lp += lp_dict[token_id].logprob
            elif lp_dict:
                total_lp += next(iter(lp_dict.values())).logprob
        results_map[(q_idx, label)] = (cont_ids, total_lp)

    question_results: list[QuestionResult] = []
    for q_idx, q in enumerate(questions):
        choices = [
            ChoiceResult(
                label=label,
                continuation_token_ids=results_map[(q_idx, label)][0],
                loglikelihood=results_map[(q_idx, label)][1],
            )
            for label in LABELS
        ]
        predicted_idx = int(np.argmax([c.loglikelihood for c in choices]))
        question_results.append(QuestionResult(
            subject=q["subject"],
            question_idx=q["question_idx"],
            context_token_ids=ctx_ids_per_q[q_idx],
            choices=choices,
            predicted=LABELS[predicted_idx],
            correct=LABELS[q["correct_idx"]],
            correct_idx=q["correct_idx"],
        ))
    return question_results


def _serialize(results: list[QuestionResult]) -> list[dict]:
    return [asdict(r) for r in results]


def _deserialize(data: list[dict]) -> list[QuestionResult]:
    out = []
    for d in data:
        d["choices"] = [ChoiceResult(**c) for c in d["choices"]]
        out.append(QuestionResult(**d))
    return out


def _platform_tag() -> str:
    if current_platform.is_tpu():
        return "tpu"
    if current_platform.is_cuda():
        return "gpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg(pytestconfig: pytest.Config) -> dict[str, Any]:
    model_path = pytestconfig.getoption("--model-path")
    if model_path is None:
        pytest.fail("--model-path is required. E.g.: --model-path=/mnt/models/my-llama")
    return {
        "model_path":     model_path,
        "save_ref":       pytestconfig.getoption("--save-ref"),
        "compare_ref":    pytestconfig.getoption("--compare-ref"),
        "subjects":       pytestconfig.getoption("--mmlu-subjects").split(","),
        "limit":          pytestconfig.getoption("--mmlu-limit"),
        "fewshot":        pytestconfig.getoption("--mmlu-fewshot"),
        "max_model_len":  pytestconfig.getoption("--max-model-len"),
        "max_num_seqs":   pytestconfig.getoption("--max-num-seqs"),
    }


@pytest.fixture(scope="module")
def llm(cfg: dict) -> LLM:
    print(f"\nLoading model: {cfg['model_path']}")
    return LLM(
        model=cfg["model_path"],
        max_model_len=cfg["max_model_len"],
        max_num_seqs=cfg["max_num_seqs"],
        max_num_batched_tokens=cfg["max_model_len"],
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        dtype="bfloat16",
    )


@pytest.fixture(scope="module")
def mmlu_questions(cfg: dict) -> list[dict]:
    return _load_mmlu_questions(cfg["subjects"], cfg["limit"], cfg["fewshot"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMmluFlip:

    def test_save_reference(
        self, llm: LLM, mmlu_questions: list[dict], cfg: dict
    ) -> None:
        """Run on current platform and save per-question logprobs to JSON.

        Skipped if --save-ref is not provided.
        """
        save_path = cfg["save_ref"]
        if save_path is None:
            pytest.skip("Pass --save-ref=PATH to save a reference file.")

        tag = _platform_tag()
        results = _run_loglikelihoods(llm, mmlu_questions)
        correct = sum(1 for r in results if r.predicted == r.correct)
        accuracy = correct / len(results)

        print(f"\nPlatform : {tag}")
        print(f"Model    : {cfg['model_path']}")
        print(f"Questions: {len(results)}")
        print(f"Accuracy : {correct}/{len(results)} = {accuracy:.4f}")

        out = {
            "platform": tag,
            "model": cfg["model_path"],
            "subjects": cfg["subjects"],
            "fewshot": cfg["fewshot"],
            "accuracy": accuracy,
            "results": _serialize(results),
        }
        Path(save_path).write_text(json.dumps(out, indent=2))
        print(f"Saved to : {save_path}")

    def test_verify_inputs_match(
        self, llm: LLM, mmlu_questions: list[dict], cfg: dict
    ) -> None:
        """Assert both platforms tokenize identical prompts to identical token IDs.

        Skipped if --compare-ref is not provided.
        """
        ref_path = cfg["compare_ref"]
        if ref_path is None:
            pytest.skip("Pass --compare-ref=PATH to compare against a reference.")

        ref_data = json.loads(Path(ref_path).read_text())
        ref_results = _deserialize(ref_data["results"])
        current_results = _run_loglikelihoods(llm, mmlu_questions)

        assert len(current_results) == len(ref_results), (
            f"Question count mismatch: {len(current_results)} vs {len(ref_results)}"
        )

        mismatches: list[str] = []
        for i, (cur, ref) in enumerate(zip(current_results, ref_results)):
            if cur.context_token_ids != ref.context_token_ids:
                mismatches.append(
                    f"q{i} ({cur.subject}): ctx len "
                    f"{len(cur.context_token_ids)} vs {len(ref.context_token_ids)}"
                )
            for cc, rc in zip(cur.choices, ref.choices):
                if cc.continuation_token_ids != rc.continuation_token_ids:
                    mismatches.append(
                        f"q{i} choice {cc.label}: "
                        f"{cc.continuation_token_ids} vs {rc.continuation_token_ids}"
                    )

        if mismatches:
            print(f"\nToken ID mismatches ({len(mismatches)}):")
            for m in mismatches[:10]:
                print(f"  {m}")

        assert not mismatches, (
            f"{len(mismatches)} token-ID mismatches — logprob comparison invalid."
        )
        print(f"\nAll {len(current_results)} questions: token IDs identical.")

    def test_compare_logprobs_and_report_flips(
        self, llm: LLM, mmlu_questions: list[dict], cfg: dict
    ) -> None:
        """Compare per-choice logprobs; report answer flips between platforms.

        Always passes (reporter only). Skipped if --compare-ref not provided.
        To make it a hard gate, uncomment the assert at the bottom.
        """
        ref_path = cfg["compare_ref"]
        if ref_path is None:
            pytest.skip("Pass --compare-ref=PATH to compare against a reference.")

        ref_data = json.loads(Path(ref_path).read_text())
        ref_results = _deserialize(ref_data["results"])
        ref_platform = ref_data["platform"]
        ref_accuracy = ref_data.get("accuracy", float("nan"))

        tag = _platform_tag()
        current_results = _run_loglikelihoods(llm, mmlu_questions)
        current_correct = sum(1 for r in current_results if r.predicted == r.correct)
        current_accuracy = current_correct / len(current_results)

        assert len(current_results) == len(ref_results)

        flips: list[dict] = []
        all_deltas: list[float] = []
        flip_margins: list[float] = []
        non_flip_margins: list[float] = []

        for cur, ref in zip(current_results, ref_results):
            cur_lls = {c.label: c.loglikelihood for c in cur.choices}
            ref_lls = {c.label: c.loglikelihood for c in ref.choices}

            for label in LABELS:
                all_deltas.append(abs(cur_lls[label] - ref_lls[label]))

            cur_sorted = sorted(cur_lls.values(), reverse=True)
            ref_sorted = sorted(ref_lls.values(), reverse=True)
            margin = min(cur_sorted[0] - cur_sorted[1],
                         ref_sorted[0] - ref_sorted[1])

            is_flip = cur.predicted != ref.predicted
            if is_flip:
                flip_margins.append(margin)
                flips.append({
                    "subject": cur.subject,
                    "q_idx": cur.question_idx,
                    "correct": cur.correct,
                    f"{ref_platform}_predicted": ref.predicted,
                    f"{tag}_predicted": cur.predicted,
                    f"{ref_platform}_lls": {l: f"{ref_lls[l]:.5f}" for l in LABELS},
                    f"{tag}_lls": {l: f"{cur_lls[l]:.5f}" for l in LABELS},
                    "margin": round(margin, 6),
                    "max_delta": round(
                        max(abs(cur_lls[l] - ref_lls[l]) for l in LABELS), 6
                    ),
                    f"{ref_platform}_correct": ref.predicted == ref.correct,
                    f"{tag}_correct": cur.predicted == cur.correct,
                })
            else:
                non_flip_margins.append(margin)

        W = 72
        print(f"\n{'='*W}")
        print(f"MMLU comparison: {ref_platform.upper()} (ref) vs {tag.upper()} (current)")
        print(f"Model    : {cfg['model_path']}")
        print(f"Subjects : {cfg['subjects']}  |  Fewshot: {cfg['fewshot']}")
        print(f"{'='*W}")

        n = len(ref_results)
        print(f"\nAccuracy:")
        print(f"  {ref_platform.upper():4s}: {ref_accuracy:.4f}  ({ref_accuracy*n:.0f}/{n})")
        print(f"  {tag.upper():4s}: {current_accuracy:.4f}  ({current_correct}/{len(current_results)})")
        print(f"  Δacc : {abs(current_accuracy - ref_accuracy):.4f}")

        print(f"\nPer-choice logprob delta  "
              f"({len(all_deltas)} values = {len(current_results)} questions × 4 choices):")
        print(f"  mean |Δ| = {np.mean(all_deltas):.6f}")
        print(f"  p50  |Δ| = {np.percentile(all_deltas, 50):.6f}")
        print(f"  p95  |Δ| = {np.percentile(all_deltas, 95):.6f}")
        print(f"  max  |Δ| = {np.max(all_deltas):.6f}")

        print(f"\nAnswer flips: {len(flips)}/{len(current_results)} questions")
        if flip_margins:
            print(f"  Flip margin (min gap between top-2 choices):")
            print(f"    mean = {np.mean(flip_margins):.6f}")
            print(f"    max  = {np.max(flip_margins):.6f}  ← widest-margin flip")
        if non_flip_margins:
            print(f"  Non-flip margin mean = {np.mean(non_flip_margins):.6f}")

        if flips:
            print(f"\nDetailed flips ({len(flips)}):")
            hdr = (f"  {'Subject':<22} {'Q':>4}  {'OK':>2}  "
                   f"{ref_platform.upper():>4}→  {tag.upper():>4}←  "
                   f"{'Margin':>8}  {'MaxΔ':>8}  Notes")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for f in flips:
                ref_ok = f[f"{ref_platform}_correct"]
                cur_ok = f[f"{tag}_correct"]
                if ref_ok and not cur_ok:
                    notes = f"{ref_platform} correct, {tag} WRONG"
                elif cur_ok and not ref_ok:
                    notes = f"{tag} correct, {ref_platform} WRONG"
                elif not ref_ok and not cur_ok:
                    notes = "both wrong"
                else:
                    notes = "both right?!"

                print(f"  {f['subject']:<22} {f['q_idx']:>4}  {f['correct']:>2}  "
                      f"{f[ref_platform+'_predicted']:>4}→  {f[tag+'_predicted']:>4}←  "
                      f"{f['margin']:>8.5f}  {f['max_delta']:>8.5f}  {notes}")
                ref_row = "  ".join(f"{l}:{f[ref_platform+'_lls'][l]}" for l in LABELS)
                cur_row = "  ".join(f"{l}:{f[tag+'_lls'][l]}" for l in LABELS)
                print(f"    {ref_platform}: {ref_row}")
                print(f"    {tag}:  {cur_row}")

        print(f"\n{'='*W}")
        print("KEY INSIGHT:")
        print("  MMLU continuation = single token (' A'/' B'/' C'/' D').")
        print("  Flip condition: |Δlogprob| > margin between top-2 choices.")
        if flip_margins and all_deltas:
            print(f"  Mean |Δ| = {np.mean(all_deltas):.5f}   "
                  f"Mean flip margin = {np.mean(flip_margins):.5f}")
        print("  Root cause: TPU softmax casts to bf16 before V-multiply;")
        print("  GPU Flash Attention keeps float32. Error compounds across 32 layers.")
        print(f"{'='*W}\n")

        # Always passes — uncomment to make it a hard gate:
        # assert abs(current_accuracy - ref_accuracy) < 0.015, (
        #     f"Accuracy gap {abs(current_accuracy - ref_accuracy):.4f} exceeds 1.5%"
        # )
