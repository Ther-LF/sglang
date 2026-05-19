"""Run KV cache quantization benchmark.

Sends prefill requests to a running sglang server and collects logprobs
for the next token prediction.

Usage:
    # Run baseline (server started WITHOUT SGLANG_KV_QUANT_BENCHMARK):
    python run_benchmark.py --port 30000 --output results_baseline.jsonl

    # Run experiment (server started WITH SGLANG_KV_QUANT_BENCHMARK=1):
    python run_benchmark.py --port 30000 --output results_quant_fp8.jsonl
"""

import json
import argparse
from pathlib import Path

import requests


def send_request(url: str, prompt: str, max_new_tokens: int = 2) -> dict:
    """Send a generate request to sglang server and get logprobs."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        "return_logprob": True,
        "top_logprobs_num": 10,
        "logprob_start_len": 0,
    }
    resp = requests.post(f"{url}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="benchmarks/kv_cache_quant/data_2k.jsonl",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    # Load data
    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    results = []
    for i, sample in enumerate(samples):
        try:
            resp = send_request(url, sample["text"])
            results.append(
                {
                    "sample_idx": i,
                    "num_input_tokens": sample["num_tokens"],
                    "response": resp,
                }
            )
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)}")
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            results.append({"sample_idx": i, "error": str(e)})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
