"""Print the per-axis floor. Run this before training anything.

    make chance

Everything here fits on train and reports on eval, exactly as a real model
would, so the numbers are directly comparable to a model's eval report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.build import load_split
from .baselines import BagOfAttributesOracle, all_blind_baselines
from .scorer import Report, compare, score


def run(data_dir: str | Path, seed: int = 0) -> tuple[list[Report], Report]:
    _, train_scenes, train_q = load_split(data_dir, "train")
    _, eval_scenes, eval_q = load_split(data_dir, "eval")

    reports: list[Report] = []
    for baseline in all_blind_baselines(seed=seed):
        preds = baseline.fit(train_q).predict(eval_q)
        reports.append(score(eval_q, preds, name=baseline.name))

    oracle = BagOfAttributesOracle(eval_scenes, seed=seed)
    oracle_report = score(eval_q, oracle.predict(eval_q), name=oracle.name)
    return reports, oracle_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-axis chance level from blind baselines.")
    ap.add_argument("--data", default="data/shapes", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    ap.add_argument("--plot", type=Path, help="also write a bar chart of the floors")
    args = ap.parse_args()

    reports, oracle = run(args.data, seed=args.seed)

    print("\nBlind baselines on eval (none of these look at the image)\n")
    print(compare(reports))
    print("\nBest blind floor per axis — this is the number to beat:\n")
    for axis in sorted(reports[0].axes):
        best = max(reports, key=lambda r: r.axes[axis].accuracy)
        a = best.axes[axis]
        lo, hi = a.ci95
        print(
            f"  {axis:<10} {a.accuracy:.3f}  [{lo:.3f}, {hi:.3f}]  "
            f"via {best.name:<10} (n={a.n})"
        )

    print("\nBag-of-attributes oracle (sees attributes, not their binding)\n")
    print(oracle.to_table(show_templates=True))
    print(
        "\n  Read this row as: perfect detection with no binding already gets\n"
        "  the count and colour axes for free. Only the binding and relation\n"
        "  gaps are evidence of anything interesting.\n"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "blind": [r.to_json() for r in reports],
                    "bag_of_attributes_oracle": oracle.to_json(),
                },
                indent=2,
            )
        )
        print(f"wrote {args.json}")

    if args.plot:
        _plot(reports, oracle, args.plot)
        print(f"wrote {args.plot}")


def _plot(reports: list[Report], oracle: Report, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    axes = sorted(reports[0].axes)
    series = reports + [oracle]
    x = np.arange(len(axes))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in enumerate(series):
        vals = [r.axes[a].accuracy if a in r.axes else 0.0 for a in axes]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=r.name)
    ax.set_xticks(x, axes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("eval accuracy")
    ax.set_title("Rung 0 floors: nothing here looks at the image (except bag-oracle)")
    ax.legend(fontsize=8, ncols=len(series))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
