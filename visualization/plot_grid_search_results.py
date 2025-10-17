import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D  # added


def main(csv_path="grid_search_results.csv", output_dir="plots", show=True):  # added show parameter
    if not os.path.isfile(csv_path):
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return

    df = pd.read_csv(csv_path)

    required_base = {"bins", "color_space", "distances"}
    if not required_base.issubset(df.columns):
        missing = required_base - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    possible_metrics = ["mapk1", "mapk5"]
    metrics = [m for m in possible_metrics if m in df.columns]
    if not metrics:
        raise ValueError("No expected metric columns found (map@1, map@5).")

    # Normalize / ensure bins sortable
    df = df.copy()
    df["bins"] = pd.to_numeric(df["bins"], errors="coerce")
    df = df.dropna(subset=["bins"])
    df = df.sort_values("bins")

    df["line_id"] = df["color_space"].astype(str) + "|" + df["distances"].astype(str)

    os.makedirs(output_dir, exist_ok=True)

    # Color cycle for top lines
    top_colors = ["#1f77b4", "#d62728"]  # blue, red
    dim_color = "#888888"

    for metric in metrics:
        # Aggregate for ranking: max value across bins per line
        ranking = (
            df.groupby("line_id")[metric]
            .max()
            .sort_values(ascending=False)
        )
        top_line_ids = ranking.index[:2].tolist()

        plt.figure(figsize=(9, 5))
        for line_id, group in df.groupby("line_id"):
            group = group.sort_values("bins")
            label = line_id.replace("|", " / ")
            if line_id in top_line_ids:
                color = top_colors[top_line_ids.index(line_id)]
                plt.plot(
                    group["bins"],
                    group[metric],
                    label=label,
                    linewidth=2.2,
                    color=color
                )
            else:
                plt.plot(
                    group["bins"],
                    group[metric],
                    color=dim_color,
                    alpha=0.35,
                    linewidth=1
                )

        # Bring top lines to front
        plt.title(f"{metric} vs Bins (color_space + distance)")
        plt.xlabel("bins")
        plt.ylabel(metric)
        plt.ylim(0, 1)  # added: fix y-axis range
        plt.xlim(0, 256)  # added: fix x-axis range
        # Legend: top lines + one combined "Others"
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(ranking.index) > len(top_line_ids):
            others_handle = Line2D(
                [0], [0],
                color=dim_color,
                alpha=0.35,
                linewidth=1,
                label="Others"
            )
            handles.append(others_handle)
            labels.append("Others")
        plt.legend(handles, labels, title="Lines", loc="best")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)

        out_path = os.path.join(output_dir, f"{metric.replace('@','_at_')}_lines.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        if show:
            plt.show()
        else:  # added: close to avoid GUI requirement
            plt.close()

if __name__ == "__main__":
    # Allow optional csv path arg
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "grid_search_results.csv"
    main(csv_arg)  # show defaults to True
