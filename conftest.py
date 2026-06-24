import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--model-path", default=None,
                     help="Local dir or HF Hub ID. E.g. /mnt/ckpt/llama-3")
    parser.addoption("--save-ref", default=None,
                     help="Path to write reference JSON (first platform).")
    parser.addoption("--compare-ref", default=None,
                     help="Path to read reference JSON (second platform).")
    parser.addoption("--mmlu-subjects", default="abstract_algebra,astronomy",
                     help="Comma-separated subjects, or 'all' for all 57 subjects.")
    parser.addoption("--mmlu-limit", default=50, type=int,
                     help="Max questions per subject. 0 = use full test split.")
    parser.addoption("--mmlu-fewshot", default=0, type=int,
                     help="Number of fewshot examples (0 or 5).")
    parser.addoption("--flip-output", default=None,
                     help="Path to write flip cases as JSON. E.g. /tmp/flips.json")
    parser.addoption("--max-model-len", default=2048, type=int,
                     help="vllm max_model_len. 1024 is enough for MMLU.")
    parser.addoption("--max-num-seqs", default=16, type=int,
                     help="vllm max_num_seqs. Reduce to 4-8 to cut KV cache HBM.")
    parser.addoption("--max-num-batched-tokens", default=None, type=int,
                     help="vllm max_num_batched_tokens. Defaults to max_model_len. "
                          "Qwen MoE on TPU may benefit from a higher value, e.g. 16384.")
    parser.addoption("--gpu-memory-utilization", default=0.9, type=float,
                     help="Fraction of HBM for KV cache (0.0-1.0). Try 0.75 on OOM.")
    parser.addoption("--tensor-parallel-size", default=1, type=int,
                     help="Number of TPU chips to shard the model across.")
