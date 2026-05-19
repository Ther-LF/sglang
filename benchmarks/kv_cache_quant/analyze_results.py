"""Analyze KV cache quantization benchmark results.

Compares baseline vs quantized logprobs and outputs metrics.

Usage:
    python analyze_results.py \
        --baseline results_baseline.jsonl \
        --experiment results_quant_fp8.jsonl
"""

import json
import argparse
import numpy as np


def extract_logprobs(result: dict):
    """Extract next-token logprobs from a sglang response.

    sglang format: output_top_logprobs is a list of steps, each step is a list
    of [logprob, token_id, token_text] triples.
    """
    resp = result.get("response", {})
    meta = resp.get("meta_info", {})

    output_top_logprobs = meta.get("output_top_logprobs", [])
    output_token_logprobs = meta.get("output_token_logprobs", [])

    if not output_top_logprobs or len(output_top_logprobs) < 2:
        return None

    # Use the SECOND decode step's logprobs (index 1)
    # The first token's logits are computed during prefill (before KV quantization)
    # The second token's logits use the quantized KV cache
    top_logprobs = output_top_logprobs[1]

    return {
        "top1_logprob": output_token_logprobs[1] if len(output_token_logprobs) > 1 else None,
        "top_logprobs": top_logprobs,  # list of [logprob, token_id, text]
    }


def compute_kl_divergence(p_list, q_list):
    """Compute KL(P || Q) over shared tokens.

    p_list, q_list: list of [logprob, token_id, text]
    """
    if not p_list or not q_list:
        return float("nan")

    p_dict = {item[1]: item[0] for item in p_list}  # token_id -> logprob
    q_dict = {item[1]: item[0] for item in q_list}

    shared_tokens = set(p_dict.keys()) & set(q_dict.keys())
    if not shared_tokens:
        return float("nan")

    kl = 0.0
    for token in shared_tokens:
        p = np.exp(p_dict[token])
        q = np.exp(q_dict[token])
        if p > 1e-10 and q > 1e-10:
            kl += p * np.log(p / q)

    return kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    baseline_results = []
    with open(args.baseline) as f:
        for line in f:
            baseline_results.append(json.loads(line))

    experiment_results = []
    with open(args.experiment) as f:
        for line in f:
            experiment_results.append(json.loads(line))

    assert len(baseline_results) == len(
        experiment_results
    ), f"Mismatch: {len(baseline_results)} vs {len(experiment_results)}"

    # Metrics
    top1_match = 0
    top1_logprob_diffs = []
    kl_divs = []
    total_valid = 0

    for base, exp in zip(baseline_results, experiment_results):
        if "error" in base or "error" in exp:
            continue

        base_lp = extract_logprobs(base)
        exp_lp = extract_logprobs(exp)

        if base_lp is None or exp_lp is None:
            continue

        total_valid += 1

        base_top = base_lp["top_logprobs"]  # list of [logprob, token_id, text]
        exp_top = exp_lp["top_logprobs"]

        if base_top and exp_top:
            # Top-1: highest logprob entry
            base_top1 = max(base_top, key=lambda x: x[0])  # [logprob, id, text]
            exp_top1 = max(exp_top, key=lambda x: x[0])

            # Token ID match
            if base_top1[1] == exp_top1[1]:
                top1_match += 1

            # Logprob diff of top-1 token
            top1_logprob_diffs.append(abs(base_top1[0] - exp_top1[0]))

            # KL divergence
            kl = compute_kl_divergence(base_top, exp_top)
            if not np.isnan(kl):
                kl_divs.append(kl)

    # Report
    print("=" * 60)
    print("KV Cache Quantization Benchmark Results")
    print("=" * 60)
    print(f"Total valid samples: {total_valid}")
    print()

    if total_valid > 0:
        print(
            f"Top-1 Token Consistency: {top1_match}/{total_valid}"
            f" = {top1_match / total_valid * 100:.1f}%"
        )
        print()

    if top1_logprob_diffs:
        diffs = np.array(top1_logprob_diffs)
        print("Top-1 Logprob Absolute Difference:")
        print(f"  Mean:   {diffs.mean():.6f}")
        print(f"  Median: {np.median(diffs):.6f}")
        print(f"  Max:    {diffs.max():.6f}")
        print(f"  P95:    {np.percentile(diffs, 95):.6f}")
        print(f"  Std:    {diffs.std():.6f}")
        print()

    if kl_divs:
        kls = np.array(kl_divs)
        print("KL Divergence (top-10 tokens):")
        print(f"  Mean:   {kls.mean():.8f}")
        print(f"  Median: {np.median(kls):.8f}")
        print(f"  Max:    {kls.max():.8f}")
        print(f"  P95:    {np.percentile(kls, 95):.8f}")
        print(f"  Std:    {kls.std():.8f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
