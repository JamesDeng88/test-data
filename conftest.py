import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--model-path", default=None,
                     help="Local dir or HF Hub ID. E.g. /mnt/ckpt/llama-3")
    parser.addoption("--save-ref", default=None,
                     help="Path to write reference JSON (first platform).")
    parser.addoption("--compare-ref", default=None,
                     help="Path to read reference JSON (second platform).")
    parser.addoption("--mmlu-subjects", default="abstract_algebra,astronomy",
                     help="Comma-separated MMLU subjects.")
    parser.addoption("--mmlu-limit", default=50, type=int,
                     help="Max questions per subject.")
    parser.addoption("--mmlu-fewshot", default=0, type=int,
                     help="Number of fewshot examples (0 or 5).")
    parser.addoption("--max-model-len", default=2048, type=int,
                     help="vllm max_model_len.")
    parser.addoption("--max-num-seqs", default=16, type=int,
                     help="vllm max_num_seqs.")
