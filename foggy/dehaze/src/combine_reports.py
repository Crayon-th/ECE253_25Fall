"""Combine NIQE result CSVs from multiple datasets and generate summary plots."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path, dataset: str) -> List[Tuple[str, str, float, str]]:
    rows: List[Tuple[str, str, float, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("filename")
            method = row.get("method")
            score = row.get("niqe")
            if not name or not method or score is None:
                continue
            try:
                score_f = float(score)
            except ValueError:
                continue
            rows.append((dataset, name, method, score_f))
    return rows


def _compute_stats(rows: List[Tuple[str, str, float, str]]) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for dataset, _, method, score in rows:
        grouped[(dataset, method)].append(score)
    stats: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    for key, values in grouped.items():
        arr = np.array(values, dtype=np.float64)
        stats[key] = (float(arr.mean()), float(arr.std()), arr.size)
    return stats


def _write_combined_csv(rows: List[Tuple[str, str, float, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "filename", "method", "niqe"])
        for dataset, name, method, score in rows:
            writer.writerow([dataset, name, method, f"{score:.6f}"])


def _write_summary(stats: Dict[Tuple[str, str], Tuple[float, float, int]], out_summary: Path) -> None:
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    datasets = sorted({k[0] for k in stats})
    methods = sorted({k[1] for k in stats})
    for dataset in datasets:
        lines.append(f"[{dataset}]")
        for method in methods:
            if (dataset, method) in stats:
                mean, std, n = stats[(dataset, method)]
                lines.append(f"  {method}: mean={mean:.4f}, std={std:.4f}, n={n}")
            else:
                lines.append(f"  {method}: no data")
        lines.append("")
    lines.append("[overall]")
    for method in methods:
        # aggregate across datasets
        vals = [stats[(ds, method)][0] for ds in datasets if (ds, method) in stats]
        counts = [stats[(ds, method)][2] for ds in datasets if (ds, method) in stats]
        if not vals:
            lines.append(f"  {method}: no data")
            continue
        weighted = np.average(np.array(vals), weights=np.array(counts))
        lines.append(f"  {method}: weighted mean={weighted:.4f} (across {sum(counts)} images)")
    with out_summary.open("w") as f:
        f.write("\n".join(lines))


def _plot_bars(stats: Dict[Tuple[str, str], Tuple[float, float, int]], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = sorted({k[0] for k in stats})
    methods = sorted({k[1] for k in stats})
    x = np.arange(len(datasets))
    width = 0.25
    offsets = {m: (i - len(methods) / 2) * width + width / 2 for i, m in enumerate(methods)}
    plt.figure(figsize=(8, 4))
    for method in methods:
        means = []
        stds = []
        for ds in datasets:
            mean, std, _ = stats.get((ds, method), (np.nan, np.nan, 0))
            means.append(mean)
            stds.append(std)
        plt.bar(x + offsets[method], means, width=width, label=method, yerr=stds, capsize=4)
    plt.xticks(x, datasets, rotation=15)
    plt.ylabel("NIQE (lower is better)")
    plt.title("NIQE by dataset and method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def _plot_overall(stats: Dict[Tuple[str, str], Tuple[float, float, int]], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({k[1] for k in stats})
    datasets = sorted({k[0] for k in stats})
    means = []
    for method in methods:
        vals = []
        weights = []
        for ds in datasets:
            if (ds, method) in stats:
                mean, _, n = stats[(ds, method)]
                vals.append(mean)
                weights.append(n)
        if vals:
            means.append(np.average(np.array(vals), weights=np.array(weights)))
        else:
            means.append(np.nan)
    plt.figure(figsize=(5, 4))
    x = np.arange(len(methods))
    plt.bar(x, means, width=0.5, color="tab:blue")
    plt.xticks(x, methods)
    plt.ylabel("NIQE (lower is better)")
    plt.title("NIQE overall weighted mean")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def _plot_box(rows: List[Tuple[str, str, float, str]], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, List[float]] = defaultdict(list)
    for _, _, method, score in rows:
        data[method].append(score)
    methods = sorted(data.keys())
    values = [data[m] for m in methods]
    plt.figure(figsize=(6, 4))
    box = plt.boxplot(values, labels=methods, showmeans=True, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    plt.ylabel("NIQE (lower is better)")
    plt.title("NIQE distribution across all datasets")
    # annotate mean ± std above each box
    for i, m in enumerate(methods, start=1):
        arr = np.array(data[m], dtype=np.float64)
        mean, std = float(arr.mean()), float(arr.std())
        plt.text(i, max(arr) if arr.size else 0, f"{mean:.2f} ± {std:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine NIQE CSVs from multiple datasets and plot summaries.")
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="List of NIQE result CSVs to combine (one per dataset).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Dataset tags corresponding to each CSV (same order as --csvs).",
    )
    parser.add_argument("--out_csv", default="niqe_results_all.csv", help="Path for combined NIQE CSV.")
    parser.add_argument("--out_summary", default="niqe_summary_all.txt", help="Path for combined NIQE summary.")
    parser.add_argument("--fig", default="figs/niqe_bar.png", help="Path to save NIQE bar plot.")
    parser.add_argument("--fig_overall", default="figs/niqe_bar_overall.png", help="Path to save overall NIQE bar plot.")
    parser.add_argument("--fig_box", default="figs/niqe_box_all.png", help="Path to save NIQE box plot (all data).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.csvs) != len(args.tags):
        raise ValueError("Number of --csvs must match number of --tags.")
    rows: List[Tuple[str, str, float, str]] = []
    for csv_path, tag in zip(args.csvs, args.tags):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV: {path}")
        rows.extend(_read_csv(path, tag))

    if not rows:
        raise RuntimeError("No rows found across provided CSVs.")

    _write_combined_csv(rows, Path(args.out_csv))
    stats = _compute_stats(rows)
    _write_summary(stats, Path(args.out_summary))
    _plot_bars(stats, Path(args.fig))
    _plot_overall(stats, Path(args.fig_overall))
    _plot_box(rows, Path(args.fig_box))
    print(f"Combined {len(rows)} rows across {len(args.csvs)} CSVs.")
    print(f"Saved combined CSV: {args.out_csv}")
    print(f"Saved summary: {args.out_summary}")
    print(f"Saved bar plot: {args.fig}")
    print(f"Saved overall bar plot: {args.fig_overall}")
    print(f"Saved box plot: {args.fig_box}")


if __name__ == "__main__":
    main()
