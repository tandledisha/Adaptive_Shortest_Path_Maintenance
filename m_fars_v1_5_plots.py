"""
M-FARS V1.5 PRESENTATION PLOTS
Reads the CSV files produced by m_fars_v1_5.py and creates six publication-ready plots.

Required files in the same folder:
  m_fars_v1_5_results.csv
  m_fars_v1_5_break_even.csv

Run:
  python m_fars_v1_5_plots.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
RESULTS_CSV = BASE / "m_fars_v1_5_results.csv"
BREAK_EVEN_CSV = BASE / "m_fars_v1_5_break_even.csv"
OUT = BASE / "plots"
OUT.mkdir(exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def i(row, key):
    return int(row[key])


def graph_sizes(rows):
    return sorted({i(r, "graph_nodes") for r in rows})


def scenarios(rows):
    return sorted({r["scenario"] for r in rows})


def mean_by(rows, key):
    values = defaultdict(list)
    for r in rows:
        values[i(r, "graph_nodes")].append(f(r, key))
    return {
        n: sum(v) / len(v)
        for n, v in values.items()
    }


def plot_1_amortized_speedup(rows):
    """Amortized speedup vs number of updates."""
    largest = max(graph_sizes(rows))
    plt.figure(figsize=(10, 6))

    for scenario in scenarios(rows):
        subset = [
            r for r in rows
            if i(r, "graph_nodes") == largest
            and r["scenario"] == scenario
        ]
        subset.sort(key=lambda r: i(r, "update_count"))
        x = [i(r, "update_count") for r in subset]
        y = [f(r, "amortized_speedup_mean") for r in subset]
        plt.plot(x, y, label=scenario)

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Number of updates on the same graph")
    plt.ylabel("Amortized speedup (Dijkstra / M-FARS)")
    plt.title(f"M-FARS V1.5: Amortized Speedup ({largest}-node graph)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / "01_amortized_speedup_vs_updates.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_2_update_time(rows):
    """Mean per-update M-FARS vs Dijkstra time by graph size."""
    sizes = graph_sizes(rows)
    mf = mean_by(rows, "mf_update_mean")
    dj = mean_by(rows, "dijkstra_update_mean")

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, [mf[n] for n in sizes], label="M-FARS")
    plt.plot(sizes, [dj[n] for n in sizes], label="Dijkstra")
    plt.xlabel("Graph size (nodes)")
    plt.ylabel("Mean update time (seconds)")
    plt.title("M-FARS vs Dijkstra Update Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / "02_mfars_vs_dijkstra_update_time.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_3_preprocessing(rows):
    """Mean one-time preprocessing cost by graph size."""
    sizes = graph_sizes(rows)
    pre = mean_by(rows, "preprocessing_mean")

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, [pre[n] for n in sizes])
    plt.xlabel("Graph size (nodes)")
    plt.ylabel("Preprocessing time (seconds)")
    plt.title("M-FARS Preprocessing Cost vs Graph Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / "03_preprocessing_cost_vs_graph_size.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_4_work_reduction(rows):
    """Mean edge-relaxation work reduction by graph size."""
    sizes = graph_sizes(rows)
    work = mean_by(rows, "work_reduction_mean")

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, [work[n] for n in sizes])
    plt.xlabel("Graph size (nodes)")
    plt.ylabel("Work reduction (%)")
    plt.title("M-FARS Work Reduction vs Graph Size")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / "04_work_reduction_vs_graph_size.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_5_break_even(break_even_rows):
    """First tested break-even update count vs graph size, one line per scenario."""
    clean = []
    for r in break_even_rows:
        if r["break_even_updates"].strip():
            clean.append(r)

    sizes = sorted({i(r, "graph_nodes") for r in clean})

    plt.figure(figsize=(10, 6))
    for scenario in sorted({r["scenario"] for r in clean}):
        subset = [
            r for r in clean
            if r["scenario"] == scenario
        ]
        subset.sort(key=lambda r: i(r, "graph_nodes"))
        x = [i(r, "graph_nodes") for r in subset]
        y = [i(r, "break_even_updates") for r in subset]
        plt.plot(x, y, label=scenario)

    plt.xlabel("Graph size (nodes)")
    plt.ylabel("Break-even update count")
    plt.title("M-FARS Break-Even Update Count vs Graph Size")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / "05_break_even_update_count_vs_graph_size.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_6_strategy_distribution(rows):
    """Strategy distribution vs update count, aggregated over graph/scenario."""
    update_counts = sorted({i(r, "update_count") for r in rows})

    # The V1.5 results CSV does not store strategy counts, so reconstruct
    # the strategy distribution from the experiment output is not possible
    # from this CSV alone. We therefore make this plot from an optional
    # strategy CSV if present.
    strategy_csv = BASE / "m_fars_v1_5_strategy_by_update.csv"
    if not strategy_csv.exists():
        print(
            "Skipping plot 6: m_fars_v1_5_strategy_by_update.csv is not present."
        )
        return None

    strategy_rows = read_csv(strategy_csv)
    strategies = [
        "SAFE_REUSE",
        "EXACT_DECREASE",
        "FULL_RECOMPUTATION",
    ]

    totals = {s: [] for s in strategies}
    for count in update_counts:
        matching = [
            r for r in strategy_rows
            if i(r, "update_count") == count
        ]
        total = sum(int(r[s]) for r in matching)
        for s in strategies:
            totals[s].append(
                100.0 * sum(int(r[s]) for r in matching) / total
                if total else 0.0
            )

    plt.figure(figsize=(10, 6))
    bottom = [0.0] * len(update_counts)

    for s in strategies:
        plt.bar(update_counts, totals[s], bottom=bottom, label=s)
        bottom = [
            a + b for a, b in zip(bottom, totals[s])
        ]

    plt.xscale("log")
    plt.xlabel("Number of updates on the same graph")
    plt.ylabel("Strategy share (%)")
    plt.title("M-FARS Strategy Distribution vs Update Count")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / "06_strategy_distribution_vs_updates.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_CSV}. Run m_fars_v1_5.py first."
        )
    if not BREAK_EVEN_CSV.exists():
        raise FileNotFoundError(
            f"Missing {BREAK_EVEN_CSV}. Run m_fars_v1_5.py first."
        )

    rows = read_csv(RESULTS_CSV)
    break_even_rows = read_csv(BREAK_EVEN_CSV)

    paths = [
        plot_1_amortized_speedup(rows),
        plot_2_update_time(rows),
        plot_3_preprocessing(rows),
        plot_4_work_reduction(rows),
        plot_5_break_even(break_even_rows),
        plot_6_strategy_distribution(rows),
    ]

    print("\nM-FARS V1.5 presentation plots created:")
    for p in paths:
        if p is not None:
            print(f"  {p}")


if __name__ == "__main__":
    main()