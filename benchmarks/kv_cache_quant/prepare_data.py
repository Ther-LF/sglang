"""Prepare C4 dataset samples for KV cache quantization benchmark.

Downloads C4 validation split, tokenizes with Qwen3.5-4B tokenizer,
filters/truncates to exactly 2048 tokens, saves 100 samples.
"""

import json
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default="benchmarks/kv_cache_quant/data_2k.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    samples = []
    for example in dataset:
        text = example["text"]
        token_ids = tokenizer.encode(text)
        if len(token_ids) >= args.seq_len:
            token_ids = token_ids[: args.seq_len]
            truncated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            samples.append(
                {
                    "text": truncated_text,
                    "token_ids": token_ids,
                    "num_tokens": len(token_ids),
                }
            )
            if len(samples) >= args.num_samples:
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
