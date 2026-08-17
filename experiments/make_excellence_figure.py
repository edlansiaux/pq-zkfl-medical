"""
Generate backdoor ASR figure (FedAvg / ℓ₂-ZKP / ZKP+median / Multi-Krum).

Reads results/backdoor_results.json and writes:
  - figures/fig_backdoor_asr.png (paper tree or --out-dir)
  - results/excellence_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "backdoor_results.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Directory for fig_backdoor_asr.png (default: ../../figures next to repo)",
    )
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    order = ["fedavg", "zkp_l2", "hybrid_zkp_median", "krum"]
    labels = {
        "fedavg": "FedAvg",
        "zkp_l2": r"ZKP $\ell_2$",
        "hybrid_zkp_median": "ZKP+median",
        "krum": "Multi-Krum",
    }
    by = {r["defense"]: r for r in data["results"]}

    names, asr, clean = [], [], []
    summary = {}
    for key in order:
        if key not in by:
            continue
        r = by[key]
        names.append(labels[key])
        asr.append(100.0 * float(r["final_asr"]))
        clean.append(100.0 * float(r["final_clean_acc"]))
        summary[key] = {
            "final_asr": r["final_asr"],
            "final_clean_acc": r["final_clean_acc"],
            "mean_asr_last5": r.get("mean_asr_last5"),
            "mean_clean_last5": r.get("mean_clean_last5"),
        }

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=160)
    b1 = ax.bar(x - w / 2, asr, w, label="ASR (%)", color="#c44e52")
    b2 = ax.bar(x + w / 2, clean, w, label="Clean acc. (%)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title(r"Trigger backdoor: $\ell_2$ alone fails; ZKP+median cuts ASR", fontsize=9)
    ax.tick_params(labelsize=7)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    fig.tight_layout()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paper_figures = os.path.abspath(os.path.join(repo_root, "..", "figures"))
    out_dir = args.out_dir or paper_figures
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "fig_backdoor_asr.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(repo_root, "results", "excellence_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"backdoor": summary, "figure": out_png}, f, indent=2)

    print(f"Wrote {out_png}")
    print(f"Wrote {summary_path}")
    print("EXCELLENCE_FIGURE_OK=1")


if __name__ == "__main__":
    main()
