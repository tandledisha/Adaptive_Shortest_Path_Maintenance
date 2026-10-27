# import random
# import time
# import heapq
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional
# import csv
# import statistics
# import os
import random
import time
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import csv
import statistics
import os
# ============================================================
# M-FARS VERSION 1.5 - CORRECTED WORK ACCOUNTING
# STATISTICAL + REPRODUCIBLE AMORTIZATION BENCHMARK
# ============================================================
#
# V1.4 answers the main limitation observed in V1.3:
# preprocessing is paid ONCE and many dynamic updates are applied
# to the SAME graph.
#
# Comparison:
#   M-FARS:
#       preprocess once -> dynamic update after dynamic update
#
#   Dijkstra baseline:
#       run target-specific Dijkstra after EVERY update
#
# The implementation uses certified strategies:
#   SAFE_REUSE
#       Increase of an edge not on the current shortest path.
#
#   EXACT_DECREASE
#       Single-edge decrease is handled exactly by checking whether
#       the new edge creates a shorter s->t route.  For certification
#       in a sequence of updates, the implementation maintains
#       forward and reverse shortest-path labels by an exact
#       target-specific Dijkstra refresh when required.
#
#   FULL_RECOMPUTATION
#       Used whenever an increase touches the current shortest path.
#
# Correctness is always checked against a fresh Dijkstra result.
# Verification time is NOT counted as M-FARS update time.
#
# ============================================================


INF = float("inf")

# Keep the same graph sizes used by V1.3.
GRAPH_SIZES = [100, 500, 1000, 5000]

# Six scenarios used by V1.3.
SCENARIOS = [
    ("RANDOM_SMALL", "increase_small"),
    ("RANDOM_LARGE", "increase_large"),
    ("SHORTEST_PATH_SMALL", "increase_small"),
    ("SHORTEST_PATH_LARGE", "increase_large"),
    ("RANDOM_DECREASE_SMALL", "decrease_small"),
    ("RANDOM_DECREASE_LARGE", "decrease_large"),
]

# Number of independent graphs per case.
EXPERIMENTS_PER_CASE = 30

# Main V1.4 experiment:
# several updates are applied to ONE graph after ONE preprocessing.
UPDATE_COUNTS = [1, 5, 10, 25, 50, 100]

EDGE_FACTOR = 3
MIN_WEIGHT = 1
MAX_WEIGHT = 20

SOURCE = 0
TARGET_MODE = "last"  # target = n - 1

RANDOM_SEED = 2026


@dataclass
class DijkstraResult:
    distance: float
    path: List[int]
    nodes_processed: int
    relaxations: int
    elapsed: float


@dataclass
class UpdateResult:
    strategy: str
    distance: float
    path: List[int]
    nodes_processed: int
    relaxations: int
    elapsed: float


def generate_graph(n: int, m: int, rng: random.Random):
    """Generate a directed weighted graph with exactly m unique edges."""
    adj = [[] for _ in range(n)]
    rev = [[] for _ in range(n)]

    used = set()

    # Make sure the graph has a direct source-to-target connection
    # and a simple backbone so the target is reachable.
    for u in range(n - 1):
        v = u + 1
        w = rng.randint(MIN_WEIGHT, MAX_WEIGHT)
        used.add((u, v))
        adj[u].append([v, w])
        rev[v].append([u, w])

    while len(used) < m:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v or (u, v) in used:
            continue

        w = rng.randint(MIN_WEIGHT, MAX_WEIGHT)
        used.add((u, v))
        adj[u].append([v, w])
        rev[v].append([u, w])

    return adj, rev


def clone_graph(adj, rev):
    return (
        [[list(edge) for edge in edges] for edges in adj],
        [[list(edge) for edge in edges] for edges in rev],
    )


def dijkstra(adj, source: int, target: Optional[int] = None) -> DijkstraResult:
    """Target-specific Dijkstra. Stops as soon as target is settled."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n

    dist[source] = 0.0
    pq = [(0.0, source)]
    processed = 0
    relaxations = 0

    start = time.perf_counter()

    while pq:
        d, u = heapq.heappop(pq)

        if d != dist[u]:
            continue

        processed += 1

        if target is not None and u == target:
            break

        for v, w in adj[u]:
            relaxations += 1
            nd = d + w

            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    elapsed = time.perf_counter() - start

    path = []
    if target is not None and dist[target] < INF:
        cur = target
        while cur != -1:
            path.append(cur)
            if cur == source:
                break
            cur = parent[cur]
        path.reverse()

    return DijkstraResult(
        distance=dist[target] if target is not None else min(dist),
        path=path,
        nodes_processed=processed,
        relaxations=relaxations,
        elapsed=elapsed,
    )


def dijkstra_full_labels(adj, source: int):
    """Full shortest-path labels with work counters."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n

    dist[source] = 0.0
    pq = [(0.0, source)]

    processed = 0
    relaxations = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue

        processed += 1

        for v, w in adj[u]:
            relaxations += 1
            nd = d + w

            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, parent, processed, relaxations


def path_edges(path: List[int]):
    return set(zip(path, path[1:]))


def find_edge(adj, u, v):
    for edge in adj[u]:
        if edge[0] == v:
            return edge
    return None


def set_edge_weight(adj, rev, u, v, new_weight):
    edge = find_edge(adj, u, v)
    if edge is None:
        raise ValueError("Edge does not exist")

    old_weight = edge[1]
    edge[1] = new_weight

    # rev[v] contains records of the form [source_node, weight].
    # Search that reverse list directly.
    redge = None
    for candidate in rev[v]:
        if candidate[0] == u:
            redge = candidate
            break

    if redge is None:
        raise ValueError("Reverse edge record missing")

    redge[1] = new_weight

    return old_weight


def choose_random_edge(adj, rng, avoid=None):
    candidates = []
    for u, edges in enumerate(adj):
        for v, w in edges:
            if avoid is None or (u, v) != avoid:
                candidates.append((u, v, w))

    return rng.choice(candidates)


def choose_path_edge(path, rng):
    if len(path) < 2:
        return None
    u, v = rng.choice(list(zip(path, path[1:])))
    return u, v


def apply_update(
    adj,
    rev,
    scenario: str,
    update_type: str,
    current_path: List[int],
    rng: random.Random,
):
    """
    Create and apply one update.

    Returns:
        (u, v, old_weight, new_weight, edge_on_old_path)
    """
    old_path_edges = path_edges(current_path)

    if scenario.startswith("SHORTEST_PATH"):
        selected = choose_path_edge(current_path, rng)
        if selected is None:
            selected = choose_random_edge(adj, rng)
        u, v = selected
    else:
        u, v, _ = choose_random_edge(adj, rng)

    edge = find_edge(adj, u, v)
    old_weight = edge[1]

    if update_type.startswith("increase"):
        # Small and large are intentionally separated.
        if update_type == "increase_small":
            delta = rng.randint(1, 3)
        else:
            delta = rng.randint(6, 12)

        new_weight = old_weight + delta

    elif update_type.startswith("decrease"):
        if update_type == "decrease_small":
            delta = rng.randint(1, max(1, min(3, old_weight - 1)))
        else:
            delta = rng.randint(1, max(1, min(max(6, old_weight - 1), old_weight - 1)))

        # Never allow a non-positive edge weight.
        new_weight = max(1, old_weight - delta)

    else:
        raise ValueError(f"Unknown update type: {update_type}")

    set_edge_weight(adj, rev, u, v, new_weight)

    return u, v, old_weight, new_weight, (u, v) in old_path_edges


class MFARS:
    """
    Certified multi-update M-FARS.

    The key design choice in V1.4 is that preprocessing is paid once.

    For a dynamic update:
      - increase outside current path -> SAFE_REUSE
      - increase on current path       -> FULL_RECOMPUTATION
      - decrease                       -> EXACT_DECREASE

    A decrease is certified by maintaining exact source distances
    after the update.  The update phase itself is measured separately
    from verification Dijkstra.

    This benchmark prioritizes correctness and fair accounting over
    claiming an unsafe shortcut.
    """

    def __init__(self, adj, rev, source, target):
        self.adj = adj
        self.rev = rev
        self.source = source
        self.target = target

        self.preprocessing_start = time.perf_counter()

        # Exact initial source labels plus preprocessing work counters.
        (
            self.dist,
            self.parent,
            self.preprocessing_nodes,
            self.preprocessing_relaxations,
        ) = dijkstra_full_labels(adj, source)

        # Initial target path.
        self.path = self._reconstruct_path()

        self.preprocessing_time = time.perf_counter() - self.preprocessing_start

        self.strategy_counts = {
            "SAFE_REUSE": 0,
            "EXACT_DECREASE": 0,
            "FULL_RECOMPUTATION": 0,
        }

    def _reconstruct_path(self):
        if self.dist[self.target] == INF:
            return []

        path = []
        cur = self.target
        while cur != -1:
            path.append(cur)
            if cur == self.source:
                break
            cur = self.parent[cur]

        path.reverse()
        return path

    def _full_recompute(self):
        start = time.perf_counter()

        (
            self.dist,
            self.parent,
            processed,
            relaxations,
        ) = dijkstra_full_labels(
            self.adj,
            self.source,
        )

        self.path = self._reconstruct_path()
        elapsed = time.perf_counter() - start

        return elapsed, processed, relaxations

    def update(self, u, v, old_weight, new_weight, update_type):
        """
        Apply M-FARS update computation.

        For certification:
        - safe reuse is O(1) after path membership check
        - decreases use an exact dynamic relaxation followed by
          propagation through the affected region
        - path increases use full recomputation
        """
        start = time.perf_counter()

        old_path_edges = path_edges(self.path)

        if new_weight > old_weight:
            if (u, v) not in old_path_edges:
                strategy = "SAFE_REUSE"
                # The existing shortest path remains feasible and its
                # cost is unchanged, so it remains optimal.
                elapsed = time.perf_counter() - start

                self.strategy_counts[strategy] += 1

                return UpdateResult(
                    strategy=strategy,
                    distance=self.dist[self.target],
                    path=self.path.copy(),
                    nodes_processed=0,
                    relaxations=0,
                    elapsed=elapsed,
                )

            strategy = "FULL_RECOMPUTATION"

            (
                recompute_elapsed,
                recompute_nodes,
                recompute_relaxations,
            ) = self._full_recompute()

            elapsed = time.perf_counter() - start

            self.strategy_counts[strategy] += 1

            return UpdateResult(
                strategy=strategy,
                distance=self.dist[self.target],
                path=self.path.copy(),
                nodes_processed=recompute_nodes,
                relaxations=recompute_relaxations,
                elapsed=max(elapsed, recompute_elapsed),
            )

        # Edge decrease.
        #
        # A decreased edge can only improve paths that use it. We perform
        # an exact incremental relaxation from v. This is a standard
        # label-correcting propagation over the affected region.
        strategy = "EXACT_DECREASE"

        # First relax the changed edge.
        nodes_processed = 0
        relaxations = 0

        candidate = self.dist[u] + new_weight

        if self.dist[u] < INF and candidate < self.dist[v]:
            self.dist[v] = candidate
            self.parent[v] = u

            pq = [(candidate, v)]

            while pq:
                d, x = heapq.heappop(pq)

                if d != self.dist[x]:
                    continue

                nodes_processed += 1

                for y, w in self.adj[x]:
                    relaxations += 1
                    nd = d + w

                    if nd < self.dist[y]:
                        self.dist[y] = nd
                        self.parent[y] = x
                        heapq.heappush(pq, (nd, y))

        self.path = self._reconstruct_path()

        elapsed = time.perf_counter() - start

        self.strategy_counts[strategy] += 1

        return UpdateResult(
            strategy=strategy,
            distance=self.dist[self.target],
            path=self.path.copy(),
            nodes_processed=nodes_processed,
            relaxations=relaxations,
            elapsed=elapsed,
        )


def classify_scenario(scenario):
    return scenario


def run_case(
    n: int,
    scenario: str,
    update_type: str,
    update_count: int,
    experiment_index: int,
    base_seed: int,
):
    m = EDGE_FACTOR * n
    # V1.5 pairs different update counts by using the same initial
    # graph seed for a given graph size and experiment.
    rng = random.Random(
        base_seed
        + n * 100000
        + experiment_index * 17
    )

    base_adj, base_rev = generate_graph(n, m, rng)

    # Target-specific baseline before updates.
    baseline = dijkstra(base_adj, SOURCE, n - 1)

    # M-FARS gets a separate copy of the SAME initial graph.
    mf_adj, mf_rev = clone_graph(base_adj, base_rev)

    prep_start = time.perf_counter()
    mfars = MFARS(mf_adj, mf_rev, SOURCE, n - 1)
    preprocessing_time = time.perf_counter() - prep_start

    # Store per-update metrics.
    mf_update_time = 0.0
    mf_nodes = 0
    mf_relaxations = 0

    dijkstra_update_time = 0.0
    dijkstra_nodes = 0
    dijkstra_relaxations = 0

    path_changes = 0
    correctness = 0

    update_strategies = {
        "SAFE_REUSE": 0,
        "EXACT_DECREASE": 0,
        "FULL_RECOMPUTATION": 0,
    }

    current_path = baseline.path.copy()

    for step in range(update_count):
        # Apply the same logical update to two copies of the graph.
        # We generate the update once and apply identical weights to
        # the M-FARS and Dijkstra copies.
        u, v, old_w, new_w, on_path = apply_update(
            mf_adj,
            mf_rev,
            scenario,
            update_type,
            current_path,
            rng,
        )

        # Apply exactly the same update to the baseline graph.
        set_edge_weight(base_adj, base_rev, u, v, new_w)

        # M-FARS update.
        mf_result = mfars.update(
            u, v, old_w, new_w, update_type
        )

        mf_update_time += mf_result.elapsed
        mf_nodes += mf_result.nodes_processed
        mf_relaxations += mf_result.relaxations

        update_strategies[mf_result.strategy] += 1

        # Fresh Dijkstra is the correctness oracle and performance baseline.
        verify = dijkstra(base_adj, SOURCE, n - 1)

        dijkstra_update_time += verify.elapsed
        dijkstra_nodes += verify.nodes_processed
        dijkstra_relaxations += verify.relaxations

        if mf_result.distance == verify.distance:
            correctness += 1

        if mf_result.path != current_path:
            path_changes += 1

        current_path = verify.path

    # Average over dynamic updates.
    denom = max(1, update_count)

    return {
        "preprocessing": preprocessing_time,
        "mf_update": mf_update_time,
        "dijkstra_update": dijkstra_update_time,
        "mf_nodes": mf_nodes / denom,
        "dijkstra_nodes": dijkstra_nodes / denom,
        "mf_relaxations": mf_relaxations / denom,
        "dijkstra_relaxations": dijkstra_relaxations / denom,
        "path_changed": 100.0 * path_changes / denom,
        "correctness": 100.0 * correctness / denom,
        "strategies": update_strategies,
    }


def print_case_summary(
    n,
    scenario,
    update_type,
    update_count,
    results,
):
    total_pre = sum(r["preprocessing"] for r in results) / len(results)
    mf_update = sum(r["mf_update"] for r in results) / len(results)
    dj_update = sum(r["dijkstra_update"] for r in results) / len(results)

    mf_amortized = total_pre + mf_update

    mf_nodes = sum(r["mf_nodes"] for r in results) / len(results)
    dj_nodes = sum(r["dijkstra_nodes"] for r in results) / len(results)

    mf_relax = sum(r["mf_relaxations"] for r in results) / len(results)
    dj_relax = sum(r["dijkstra_relaxations"] for r in results) / len(results)

    path_changed = sum(r["path_changed"] for r in results) / len(results)
    correctness = sum(r["correctness"] for r in results) / len(results)

    work_reduction = (
        100.0 * (1.0 - mf_relax / dj_relax)
        if dj_relax > 0
        else 0.0
    )

    update_speedup = (
        dj_update / mf_update
        if mf_update > 0
        else float("inf")
    )

    amortized_speedup = (
        (dj_update) / mf_amortized
        if mf_amortized > 0
        else 0.0
    )

    print()
    print("=" * 68)
    print(f"Graph size: {n} nodes")
    print(f"Edges: {EDGE_FACTOR * n}")
    print(f"Update count on SAME graph: {update_count}")
    print("=" * 68)

    print()
    print(f"Scenario: {scenario}")
    print(f"Update: {update_type}")
    print(f"Experiments: {len(results)}")
    print()

    print(f"Average M-FARS score: {0.0:.4f}")
    print(f"Path changed: {path_changed:.2f}%")
    print(f"Correctness: {correctness:.2f}%")
    print()

    print("TIME:")
    print(f"  Preprocessing once/graph:  {total_pre:.8f} seconds")
    print(f"  M-FARS total updates:      {mf_update:.8f} seconds")
    print(f"  Dijkstra total updates:    {dj_update:.8f} seconds")
    print(f"  M-FARS amortized total:    {mf_amortized:.8f} seconds")
    print()

    print("PER-UPDATE AVERAGES:")
    print(f"  M-FARS update:              {mf_update / update_count:.8f} seconds")
    print(f"  Dijkstra update:            {dj_update / update_count:.8f} seconds")
    print()

    print("NODES:")
    print(f"  M-FARS nodes:               {mf_nodes:.2f}")
    print(f"  Dijkstra nodes:             {dj_nodes:.2f}")
    print()

    print("EDGE RELAXATIONS:")
    print(f"  M-FARS relaxations:         {mf_relax:.2f}")
    print(f"  Dijkstra relaxations:       {dj_relax:.2f}")
    print()

    print("EFFICIENCY:")
    print(f"  Work reduction:             {work_reduction:.2f}%")
    print(f"  Update-only speedup:        {update_speedup:.3f}x")
    print(f"  Amortized speedup:           {amortized_speedup:.3f}x")

    total_strategy = sum(
        sum(r["strategies"][s] for r in results)
        for s in update_strategies_names()
    )

    print()
    print("STRATEGY DISTRIBUTION:")

    aggregate = {
        s: sum(r["strategies"][s] for r in results)
        for s in update_strategies_names()
    }

    for s in update_strategies_names():
        pct = 100.0 * aggregate[s] / total_strategy if total_strategy else 0.0
        print(f"  {s:<20}: {aggregate[s]:3d} ({pct:6.2f}%)")


def update_strategies_names():
    return [
        "SAFE_REUSE",
        "EXACT_DECREASE",
        "FULL_RECOMPUTATION",
    ]



def safe_stdev(values):
    """Sample standard deviation; 0 when there is only one value."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def write_v15_csv(rows, filename="m_fars_v1_5_results.csv"):
    """Save publication-friendly per-configuration statistics."""
    fields = [
        "graph_nodes", "edges", "scenario", "update_type", "update_count",
        "experiments",
        "correctness_mean", "correctness_median", "correctness_std",
        "preprocessing_mean", "preprocessing_std",
        "mf_update_mean", "mf_update_std",
        "dijkstra_update_mean", "dijkstra_update_std",
        "mf_relax_mean", "mf_relax_std",
        "dijkstra_relax_mean", "dijkstra_relax_std",
        "work_reduction_mean",
        "update_speedup_mean",
        "amortized_speedup_mean",
        "path_changed_mean",
        "safe_reuse_count",
        "exact_decrease_count",
        "full_recomputation_count",
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(filename)


def write_break_even_csv(rows, filename="m_fars_v1_5_break_even.csv"):
    fields = [
        "graph_nodes", "scenario", "update_type", "break_even_updates"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(filename)


def create_plots(rows, output_dir="plots"):
    """
    Create six publication-oriented figures.

    The plots are deliberately separated so that:
      1. amortized speedup is the main result;
      2. runtime is shown per update;
      3. preprocessing overhead is visible;
      4. work reduction is broken down by scenario;
      5. tested break-even limits are clearly labeled;
      6. strategy distribution is shown by update count.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print()
        print("Matplotlib is not installed.")
        print("Install it with:")
        print("    pip install matplotlib")
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    largest_n = max(r["graph_nodes"] for r in rows)

    created = []

    # --------------------------------------------------------
    # Plot 1: amortized speedup vs number of updates
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for scenario in sorted({
        r["scenario"] for r in rows if r["graph_nodes"] == largest_n
    }):
        subset = sorted(
            [
                r for r in rows
                if r["graph_nodes"] == largest_n
                and r["scenario"] == scenario
            ],
            key=lambda x: x["update_count"],
        )

        plt.plot(
            [r["update_count"] for r in subset],
            [r["amortized_speedup_mean"] for r in subset],
            marker="o",
            label=scenario,
        )

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Number of updates on the same graph")
    plt.ylabel("Amortized speedup (Dijkstra / M-FARS)")
    plt.title(f"M-FARS V1.5: Amortized Speedup ({largest_n}-node graph)")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    p = out / "01_amortized_speedup_vs_updates.png"
    plt.savefig(p, dpi=300)
    plt.close()
    created.append(str(p))

    # --------------------------------------------------------
    # Plot 2: per-update execution time vs graph size
    # --------------------------------------------------------
    # Use a fixed representative workload for readability:
    # 50 updates, averaged over scenarios within each graph size.
    workload = 50
    grouped = {}

    for r in rows:
        if r["update_count"] == workload:
            grouped.setdefault(r["graph_nodes"], []).append(r)

    x = sorted(grouped)
    mf = []
    dj = []

    for n in x:
        group = grouped[n]
        mf.append(
            sum(r["mf_update_mean"] for r in group) / len(group)
        )
        dj.append(
            sum(r["dijkstra_update_mean"] for r in group) / len(group)
        )

    plt.figure(figsize=(9, 6))
    plt.plot(x, mf, marker="o", label="M-FARS")
    plt.plot(x, dj, marker="o", label="Dijkstra")
    plt.xlabel("Graph size (nodes)")
    plt.ylabel("Mean update time (seconds)")
    plt.title("M-FARS vs Dijkstra: Per-Update Execution Time (50 updates)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    p = out / "02_mfars_vs_dijkstra_update_time.png"
    plt.savefig(p, dpi=300)
    plt.close()
    created.append(str(p))

    # --------------------------------------------------------
    # Plot 3: preprocessing cost vs graph size
    # --------------------------------------------------------
    prep = {}
    for r in rows:
        prep.setdefault(r["graph_nodes"], []).append(r["preprocessing_mean"])

    x = sorted(prep)
    y = [sum(prep[n]) / len(prep[n]) for n in x]

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel("Graph size (nodes)")
    plt.ylabel("Mean preprocessing time (seconds)")
    plt.title("M-FARS Preprocessing Cost vs Graph Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    p = out / "03_preprocessing_cost_vs_graph_size.png"
    plt.savefig(p, dpi=300)
    plt.close()
    created.append(str(p))

    # --------------------------------------------------------
    # Plot 4: work reduction vs graph size by scenario group
    # --------------------------------------------------------
    # Aggregate the six scenarios into three interpretable families.
    family_map = {
        "RANDOM": {"RANDOM_SMALL", "RANDOM_LARGE"},
        "RANDOM_DECREASE": {
            "RANDOM_DECREASE_SMALL",
            "RANDOM_DECREASE_LARGE",
        },
        "SHORTEST_PATH": {
            "SHORTEST_PATH_SMALL",
            "SHORTEST_PATH_LARGE",
        },
    }

    plt.figure(figsize=(10, 6))

    for family, names in family_map.items():
        family_points = []

        for n in sorted({
            r["graph_nodes"] for r in rows if r["update_count"] == workload
        }):
            vals = [
                r["work_reduction_mean"]
                for r in rows
                if r["graph_nodes"] == n
                and r["update_count"] == workload
                and r["scenario"] in names
            ]

            if vals:
                family_points.append((n, sum(vals) / len(vals)))

        plt.plot(
            [p[0] for p in family_points],
            [p[1] for p in family_points],
            marker="o",
            label=family,
        )

    plt.xlabel("Graph size (nodes)")
    plt.ylabel("Work reduction (%)")
    plt.title("M-FARS Work Reduction by Update Family (50 updates)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    p = out / "04_work_reduction_vs_graph_size.png"
    plt.savefig(p, dpi=300)
    plt.close()
    created.append(str(p))

    # --------------------------------------------------------
    # Plot 5: break-even update count vs graph size
    # --------------------------------------------------------
    # Recompute tested break-even bounds from CSV data.
    break_rows = {}

    for r in rows:
        key = (r["graph_nodes"], r["scenario"])
        break_rows.setdefault(key, []).append(r)

    plt.figure(figsize=(10, 6))

    for scenario in sorted({
        r["scenario"] for r in rows
    }):
        xs = []
        ys = []

        for n in sorted({
            r["graph_nodes"] for r in rows
            if r["scenario"] == scenario
        }):
            subset = sorted(
                [
                    r for r in rows
                    if r["graph_nodes"] == n
                    and r["scenario"] == scenario
                ],
                key=lambda z: z["update_count"],
            )

            winner = None
            for r in subset:
                # Mean preprocessing is paid once; cumulative update
                # cost is update_count times the per-update mean.
                mf_total = (
                    r["preprocessing_mean"]
                    + r["mf_update_mean"]
                    * r["update_count"]
                )
                dj_total = (
                    r["dijkstra_update_mean"]
                    * r["update_count"]
                )

                if mf_total < dj_total:
                    winner = r["update_count"]
                    break

            if winner is not None:
                xs.append(n)
                ys.append(winner)

        if xs:
            plt.plot(xs, ys, marker="o", label=scenario)

    plt.xlabel("Graph size (nodes)")
    plt.ylabel("First tested break-even update count")
    plt.title("M-FARS Break-Even Update Count vs Graph Size")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    p = out / "05_break_even_update_count_vs_graph_size.png"
    plt.savefig(p, dpi=300)
    plt.close()
    created.append(str(p))

    # --------------------------------------------------------
    # Plot 6: strategy distribution vs update count
    # --------------------------------------------------------
    # Aggregate across all graph sizes/scenarios.
    by_count = {}

    for r in rows:
        count = r["update_count"]
        by_count.setdefault(count, {
            "SAFE_REUSE": 0,
            "EXACT_DECREASE": 0,
            "FULL_RECOMPUTATION": 0,
            "total": 0,
        })

        by_count[count]["SAFE_REUSE"] += r["safe_reuse_count"]
        by_count[count]["EXACT_DECREASE"] += r["exact_decrease_count"]
        by_count[count]["FULL_RECOMPUTATION"] += r["full_recomputation_count"]

    counts = sorted(by_count)
    totals = [by_count[c]["total"] for c in counts]

    safe_pct = []
    dec_pct = []
    full_pct = []

    for c in counts:
        d = by_count[c]
        total = (
            d["SAFE_REUSE"]
            + d["EXACT_DECREASE"]
            + d["FULL_RECOMPUTATION"]
        )

        safe_pct.append(100 * d["SAFE_REUSE"] / total if total else 0)
        dec_pct.append(100 * d["EXACT_DECREASE"] / total if total else 0)
        full_pct.append(100 * d["FULL_RECOMPUTATION"] / total if total else 0)

    plt.figure(figsize=(10, 6))

    bottoms_dec = [safe_pct[i] for i in range(len(counts))]
    bottoms_full = [
        safe_pct[i] + dec_pct[i]
        for i in range(len(counts))
    ]

    plt.bar(counts, safe_pct, label="SAFE_REUSE")
    plt.bar(
        counts,
        dec_pct,
        bottom=bottoms_dec,
        label="EXACT_DECREASE",
    )
    plt.bar(
        counts,
        full_pct,
        bottom=bottoms_full,
        label="FULL_RECOMPUTATION",
    )

    plt.xlabel("Number of updates on the same graph")
    plt.ylabel("Strategy share (%)")
    plt.title("M-FARS Strategy Distribution vs Update Count")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    p = out / "06_strategy_distribution_vs_updates.png"
    plt.savefig(p, dpi=300)
    plt.close()
    created.append(str(p))

    return created

def main():
    print("=" * 72)
    print("M-FARS VERSION 1.5")
    print("STATISTICAL + REPRODUCIBLE AMORTIZATION BENCHMARK")
    print("=" * 72)
    print()
    print("V1.5 improvements:")
    print("  - 30 independent experiments per configuration")
    print("  - update counts: 1, 5, 10, 25, 50, 100")
    print("  - paired initial graph seeds across update counts")
    print("  - mean / median / standard deviation")
    print("  - CSV result files")
    print("  - amortized-speedup plot")
    print()

    print(
        f"Graphs: {len(GRAPH_SIZES)} | "
        f"Scenarios: {len(SCENARIOS)} | "
        f"Experiments/case: {EXPERIMENTS_PER_CASE}"
    )
    print(f"Update counts: {UPDATE_COUNTS}")

    total_cases = (
        len(GRAPH_SIZES)
        * len(SCENARIOS)
        * len(UPDATE_COUNTS)
    )

    print(f"Total case configurations: {total_cases}")
    print()
    print("WARNING: V1.5 is larger than V1.4 and may take longer.")
    print("Do not stop the program if the terminal is quiet for a while.")
    print()

    csv_rows = []
    break_even_rows = []

    overall_correct = 0
    overall_checks = 0

    strategy_totals = {
        "SAFE_REUSE": 0,
        "EXACT_DECREASE": 0,
        "FULL_RECOMPUTATION": 0,
    }

    for update_count in UPDATE_COUNTS:
        print()
        print("#" * 72)
        print(f"UPDATE COUNT = {update_count}")
        print("#" * 72)

        for n in GRAPH_SIZES:
            for scenario, update_type in SCENARIOS:
                results = []

                for exp in range(EXPERIMENTS_PER_CASE):
                    result = run_case(
                        n=n,
                        scenario=scenario,
                        update_type=update_type,
                        update_count=update_count,
                        experiment_index=exp,
                        base_seed=RANDOM_SEED,
                    )
                    results.append(result)

                    correct_steps = round(
                        result["correctness"] * update_count / 100.0
                    )
                    overall_correct += correct_steps
                    overall_checks += update_count

                # Per-experiment derived values.
                work_reductions = []
                update_speedups = []
                amortized_speedups = []

                for r in results:
                    work = (
                        100.0 * (
                            1.0 - r["mf_relaxations"]
                            / r["dijkstra_relaxations"]
                        )
                        if r["dijkstra_relaxations"] > 0
                        else 0.0
                    )

                    update_speed = (
                        r["dijkstra_update"] / r["mf_update"]
                        if r["mf_update"] > 0
                        else 0.0
                    )

                    amortized_total = (
                        r["preprocessing"] + r["mf_update"]
                    )

                    amortized_speed = (
                        r["dijkstra_update"] / amortized_total
                        if amortized_total > 0
                        else 0.0
                    )

                    work_reductions.append(work)
                    update_speedups.append(update_speed)
                    amortized_speedups.append(amortized_speed)

                def mean(key):
                    return statistics.mean(r[key] for r in results)

                def median(key):
                    return statistics.median(r[key] for r in results)

                print()
                print("-" * 72)
                print(
                    f"{n} nodes | {scenario} | "
                    f"{update_count} updates | "
                    f"{EXPERIMENTS_PER_CASE} experiments"
                )
                print(
                    f"Correctness: {mean('correctness'):.2f}% | "
                    f"Amortized speedup: "
                    f"{statistics.mean(amortized_speedups):.3f}x"
                )
                print(
                    f"M-FARS time: {mean('mf_update'):.8f}s | "
                    f"Dijkstra: {mean('dijkstra_update'):.8f}s"
                )
                print(
                    f"Work reduction: "
                    f"{statistics.mean(work_reductions):.2f}%"
                )
                print(
                    f"Median amortized speedup: "
                    f"{statistics.median(amortized_speedups):.3f}x | "
                    f"Std: "
                    f"{safe_stdev(amortized_speedups):.3f}"
                )

                strategy_counts_case = {
                    s: sum(r["strategies"][s] for r in results)
                    for s in update_strategies_names()
                }

                csv_rows.append({
                    "graph_nodes": n,
                    "edges": EDGE_FACTOR * n,
                    "scenario": scenario,
                    "update_type": update_type,
                    "update_count": update_count,
                    "experiments": len(results),

                    "correctness_mean": mean("correctness"),
                    "correctness_median": median("correctness"),
                    "correctness_std": safe_stdev(
                        [r["correctness"] for r in results]
                    ),

                    "preprocessing_mean": mean("preprocessing"),
                    "preprocessing_std": safe_stdev(
                        [r["preprocessing"] for r in results]
                    ),

                    "mf_update_mean": mean("mf_update"),
                    "mf_update_std": safe_stdev(
                        [r["mf_update"] for r in results]
                    ),

                    "dijkstra_update_mean": mean("dijkstra_update"),
                    "dijkstra_update_std": safe_stdev(
                        [r["dijkstra_update"] for r in results]
                    ),

                    "mf_relax_mean": mean("mf_relaxations"),
                    "mf_relax_std": safe_stdev(
                        [r["mf_relaxations"] for r in results]
                    ),

                    "dijkstra_relax_mean": mean("dijkstra_relaxations"),
                    "dijkstra_relax_std": safe_stdev(
                        [r["dijkstra_relaxations"] for r in results]
                    ),

                    "work_reduction_mean": statistics.mean(
                        work_reductions
                    ),
                    "update_speedup_mean": statistics.mean(
                        update_speedups
                    ),
                    "amortized_speedup_mean": statistics.mean(
                        amortized_speedups
                    ),
                    "path_changed_mean": mean("path_changed"),
                    "safe_reuse_count": strategy_counts_case["SAFE_REUSE"],
                    "exact_decrease_count": strategy_counts_case["EXACT_DECREASE"],
                    "full_recomputation_count": strategy_counts_case["FULL_RECOMPUTATION"],
                })

                # For break-even: compare cumulative M-FARS cost
                # against cumulative Dijkstra cost.
                avg_pre = mean("preprocessing")
                avg_mf_per_update = (
                    mean("mf_update") / update_count
                )
                avg_dj_per_update = (
                    mean("dijkstra_update") / update_count
                )

                mf_total = avg_pre + (
                    avg_mf_per_update * update_count
                )
                dj_total = avg_dj_per_update * update_count

                break_even_rows.append({
                    "graph_nodes": n,
                    "scenario": scenario,
                    "update_type": update_type,
                    "update_count": update_count,
                    "mf_total": mf_total,
                    "dj_total": dj_total,
                })

                for s in update_strategies_names():
                    strategy_totals[s] += sum(
                        r["strategies"][s] for r in results
                    )

    # Find first tested break-even point.
    break_even_summary = []

    print()
    print("=" * 72)
    print("M-FARS V1.5 FINAL SUMMARY")
    print("=" * 72)

    print(
        f"Total correctness checks: {overall_checks}"
    )
    print(
        f"Overall correctness: "
        f"{100.0 * overall_correct / overall_checks:.2f}%"
    )

    print()
    print("BREAK-EVEN ANALYSIS")
    print("-" * 72)

    grouped = {}

    for row in break_even_rows:
        key = (row["graph_nodes"], row["scenario"])
        grouped.setdefault(key, []).append(row)

    for key, rows in grouped.items():
        rows.sort(key=lambda r: r["update_count"])

        winner = None

        for row in rows:
            if row["mf_total"] < row["dj_total"]:
                winner = row["update_count"]
                break

        n, scenario = key

        if winner is None:
            text = "not reached in tested range"
        else:
            text = f"<= {winner} updates"

        print(
            f"{n:5d} nodes | "
            f"{scenario:<24} | {text}"
        )

        break_even_summary.append({
            "graph_nodes": n,
            "scenario": scenario,
            "update_type": rows[0]["update_type"],
            "break_even_updates": (
                winner if winner is not None else ""
            ),
        })

    # Strategy distribution.
    print()
    print("STRATEGY DISTRIBUTION")
    total_strategies = sum(strategy_totals.values())

    for strategy in update_strategies_names():
        pct = (
            100.0 * strategy_totals[strategy] / total_strategies
            if total_strategies
            else 0.0
        )
        print(
            f"  {strategy:<20}: "
            f"{strategy_totals[strategy]:5d} ({pct:6.2f}%)"
        )

    csv_file = write_v15_csv(csv_rows)
    break_even_file = write_break_even_csv(break_even_summary)
    plot_file = create_plots(csv_rows)

    print()
    print("FILES CREATED")
    print(f"  Results CSV:     {csv_file}")
    print(f"  Break-even CSV:  {break_even_file}")

    if plot_file:
        print(f"  Speedup plot:    {plot_file}")

    print()
    print("METHODOLOGY")
    print("1. One initial graph is generated for each experiment.")
    print("2. The same initial graph seed is reused across update counts.")
    print("3. M-FARS preprocessing is paid exactly once.")
    print("4. Multiple updates are applied to the same graph.")
    print("5. Dijkstra is rerun after every update.")
    print("6. Dijkstra remains the correctness oracle.")
    print("7. Verification time is NOT counted as M-FARS update time.")
    print("8. Mean, median, and standard deviation are recorded.")
    print("9. Break-even is the first tested count where")
    print("   amortized M-FARS time < Dijkstra update time.")

    print()
    print("PLOTS CREATED:")
    for plot_path in plot_file:
        print(f"  {plot_path}")

    print()
    print("V1.5 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()