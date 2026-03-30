"""
Compare quality metrics between two result files.

Usage:
    python scripts/compare_results.py \
        --random outputs/results.json \
        --low outputs/results_low.json
"""
import json
import argparse
import statistics
from collections import defaultdict


METRICS = ["ppl", "seq_rep_1", "seq_rep_2", "seq_rep_3", "z_score"]
METRIC_LABELS = {
    "ppl": "PPL",
    "seq_rep_1": "Seq-Rep-1",
    "seq_rep_2": "Seq-Rep-2",
    "seq_rep_3": "Seq-Rep-3",
    "z_score": "Z-Score",
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(records, metric):
    values = [r[metric] for r in records if r.get(metric) is not None]
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std, len(values)


def print_table(rows, headers, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))


def compare(random_path, low_path):
    random_records = load_jsonl(random_path)
    low_records = load_jsonl(low_path)

    print(f"\nLoaded {len(random_records)} records from: {random_path}")
    print(f"Loaded {len(low_records)} records from: {low_path}\n")

    datasets = sorted(set(r["dataset_name"] for r in random_records))

    # --- Overall comparison ---
    print("=" * 72)
    print("OVERALL COMPARISON")
    print("=" * 72)

    headers = ["Metric", "Random (mean±std)", "LowConf (mean±std)", "Δ (Low−Rand)"]
    col_widths = [12, 22, 22, 14]
    rows = []
    for metric in METRICS:
        r_mean, r_std, r_n = compute_stats(random_records, metric)
        l_mean, l_std, l_n = compute_stats(low_records, metric)
        if r_mean is None or l_mean is None:
            continue
        delta = l_mean - r_mean
        rows.append([
            METRIC_LABELS[metric],
            f"{r_mean:.4f} ± {r_std:.4f}",
            f"{l_mean:.4f} ± {l_std:.4f}",
            f"{delta:+.4f}",
        ])
    print_table(rows, headers, col_widths)

    # --- Per-dataset breakdown ---
    for dataset in datasets:
        print(f"\n{'=' * 72}")
        print(f"DATASET: {dataset}")
        print("=" * 72)
        r_sub = [r for r in random_records if r["dataset_name"] == dataset]
        l_sub = [r for r in low_records if r["dataset_name"] == dataset]
        rows = []
        for metric in METRICS:
            r_mean, r_std, r_n = compute_stats(r_sub, metric)
            l_mean, l_std, l_n = compute_stats(l_sub, metric)
            if r_mean is None or l_mean is None:
                continue
            delta = l_mean - r_mean
            rows.append([
                METRIC_LABELS[metric],
                f"{r_mean:.4f} ± {r_std:.4f}  (n={r_n})",
                f"{l_mean:.4f} ± {l_std:.4f}  (n={l_n})",
                f"{delta:+.4f}",
            ])
        print_table(rows, headers, [12, 26, 26, 14])

    # --- Paper reference ---
    print(f"\n{'=' * 72}")
    print("PAPER REFERENCE (Table 1, ICLR 2026 — LLaDA-8B, C={-1}, δ=4, random)")
    print("=" * 72)
    print(f"  {'Metric':<14} {'Paper value'}")
    print(f"  {'-'*14} {'-'*12}")
    print(f"  {'log(PPL)':<14} 1.90")
    print(f"  {'(PPL)':<14} ~{2.718**1.90:.2f}  (exp(1.90))")
    print()
    r_ppl_mean, _, _ = compute_stats(random_records, "ppl")
    l_ppl_mean, _, _ = compute_stats(low_records, "ppl")
    if r_ppl_mean is not None:
        import math
        print(f"  Our random remasking  PPL = {r_ppl_mean:.4f}  (log PPL = {math.log(r_ppl_mean):.4f})")
    if l_ppl_mean is not None:
        import math
        print(f"  Our low-conf remasking PPL = {l_ppl_mean:.4f}  (log PPL = {math.log(l_ppl_mean):.4f})")
    print()
    print("Note: Paper uses δ=4; our runs use δ=2, which may explain PPL differences.")


def main():
    parser = argparse.ArgumentParser(description="Compare quality metrics between two result files")
    parser.add_argument("--random", default="outputs/results.json", help="Path to random-remasking results")
    parser.add_argument("--low", default="outputs/results_low.json", help="Path to low-confidence remasking results")
    args = parser.parse_args()
    compare(args.random, args.low)


if __name__ == "__main__":
    main()
