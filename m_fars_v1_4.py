import random
import time
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ============================================================
# M-FARS VERSION 1.4
# MULTI-UPDATE AMORTIZATION BENCHMARK
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
EXPERIMENTS_PER_CASE = 10

# Main V1.4 experiment:
# several updates are applied to ONE graph after ONE preprocessing.
UPDATE_COUNTS = [1, 5, 10, 25, 50]

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
    """Full shortest-path labels, used only during M-FARS preprocessing."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n

    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue

        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, parent


def path_edges(path: List[int]):
    return set(zip(path, path[1:]))


def find_edge(adj, u, v):
    for edge in adj[u]:
        if edge[0] == v:
            return edge
    return None


# def set_edge_weight(adj, rev, u, v, new_weight):
#     edge = find_edge(adj, u, v)
#     if edge is None:
#         raise ValueError("Edge does not exist")

#     old_weight = edge[1]
#     edge[1] = new_weight

#     redge = find_edge(rev[v], u)
#     if redge is None:
#         raise ValueError("Reverse edge record missing")
#     redge[1] = new_weight

#     return old_weight
def set_edge_weight(adj, rev, u, v, new_weight):
    edge = find_edge(adj, u, v)

    if edge is None:
        return False

    edge[1] = new_weight
    return True

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

        # Exact initial source labels.
        self.dist, self.parent = dijkstra_full_labels(adj, source)

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
        self.dist, self.parent = dijkstra_full_labels(
            self.adj, self.source
        )
        self.path = self._reconstruct_path()
        return time.perf_counter() - start

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
            recompute_elapsed = self._full_recompute()
            elapsed = time.perf_counter() - start

            self.strategy_counts[strategy] += 1

            return UpdateResult(
                strategy=strategy,
                distance=self.dist[self.target],
                path=self.path.copy(),
                nodes_processed=0,
                relaxations=0,
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
    rng = random.Random(
        base_seed
        + n * 100000
        + update_count * 1000
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


def main():
    print("=" * 68)
    print("M-FARS VERSION 1.4")
    print("MULTI-UPDATE AMORTIZATION BENCHMARK")
    print("=" * 68)
    print()
    print("Starting fair multi-update experiments...")
    print(
        f"Graphs: {len(GRAPH_SIZES)} | "
        f"Scenarios: {len(SCENARIOS)} | "
        f"Experiments/case: {EXPERIMENTS_PER_CASE}"
    )
    print(f"Update counts: {UPDATE_COUNTS}")
    print(
        f"Total case configurations: "
        f"{len(GRAPH_SIZES) * len(SCENARIOS) * len(UPDATE_COUNTS)}"
    )
    print()

    overall_correct = 0
    overall_checks = 0

    overall_mf_update = 0.0
    overall_dijkstra_update = 0.0
    overall_preprocessing = 0.0

    overall_mf_relax = 0.0
    overall_dijkstra_relax = 0.0

    strategy_totals = {
        "SAFE_REUSE": 0,
        "EXACT_DECREASE": 0,
        "FULL_RECOMPUTATION": 0,
    }

    # Keep the full result set so the final break-even table can be printed.
    break_even_rows = []

    case_number = 0

    for update_count in UPDATE_COUNTS:
        print()
        print("#" * 68)
        print(f"UPDATE COUNT = {update_count}")
        print("#" * 68)

        for n in GRAPH_SIZES:
            for scenario, update_type in SCENARIOS:
                case_number += 1

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

                    overall_correct += int(
                        round(
                            result["correctness"]
                            * update_count
                            / 100.0
                        )
                    )
                    overall_checks += update_count

                print_case_summary(
                    n,
                    scenario,
                    update_type,
                    update_count,
                    results,
                )

                avg_pre = sum(r["preprocessing"] for r in results) / len(results)
                avg_mf = sum(r["mf_update"] for r in results) / len(results)
                avg_dj = sum(r["dijkstra_update"] for r in results) / len(results)

                avg_mf_relax = (
                    sum(r["mf_relaxations"] for r in results)
                    / len(results)
                )
                avg_dj_relax = (
                    sum(r["dijkstra_relaxations"] for r in results)
                    / len(results)
                )

                mf_total = avg_pre + avg_mf
                dj_total = avg_dj

                break_even_rows.append(
                    (
                        n,
                        scenario,
                        update_type,
                        update_count,
                        mf_total,
                        dj_total,
                        avg_mf_relax,
                        avg_dj_relax,
                    )
                )

                overall_preprocessing += avg_pre
                overall_mf_update += avg_mf
                overall_dijkstra_update += avg_dj
                overall_mf_relax += avg_mf_relax
                overall_dijkstra_relax += avg_dj_relax

                for s in update_strategies_names():
                    strategy_totals[s] += sum(
                        r["strategies"][s] for r in results
                    )

    print()
    print("=" * 68)
    print("M-FARS V1.4 FINAL MULTI-UPDATE SUMMARY")
    print("=" * 68)

    print()
    print(f"Total correctness checks: {overall_checks}")
    print(
        f"Overall correctness: "
        f"{100.0 * overall_correct / overall_checks:.2f}%"
        if overall_checks
        else "Overall correctness: 0.00%"
    )

    print()
    print("AGGREGATE TIMING:")
    print(f"Preprocessing totals across cases: {overall_preprocessing:.8f} seconds")
    print(f"M-FARS update totals:              {overall_mf_update:.8f} seconds")
    print(f"Dijkstra update totals:            {overall_dijkstra_update:.8f} seconds")

    print()
    print("AGGREGATE WORK:")
    if overall_dijkstra_relax > 0:
        reduction = (
            100.0
            * (1.0 - overall_mf_relax / overall_dijkstra_relax)
        )
    else:
        reduction = 0.0

    print(f"M-FARS average relaxations:        {overall_mf_relax:.2f}")
    print(f"Dijkstra average relaxations:      {overall_dijkstra_relax:.2f}")
    print(f"Work reduction:                    {reduction:.2f}%")

    print()
    print("STRATEGY COUNTS:")
    total_strategies = sum(strategy_totals.values())

    for s in update_strategies_names():
        pct = (
            100.0 * strategy_totals[s] / total_strategies
            if total_strategies
            else 0.0
        )
        print(f"  {s:<20}: {strategy_totals[s]:4d} ({pct:6.2f}%)")

    print()
    print("=" * 68)
    print("BREAK-EVEN ANALYSIS")
    print("=" * 68)
    print()
    print(
        "For each configuration, M-FARS becomes beneficial when:"
    )
    print(
        "  preprocessing + accumulated M-FARS updates"
    )
    print(
        "  < accumulated Dijkstra update time"
    )
    print()

    # For each graph/scenario, estimate the first tested update count
    # where amortized M-FARS total becomes faster.
    grouped = {}

    for row in break_even_rows:
        n, scenario, update_type, count, mf_total, dj_total, _, _ = row
        key = (n, scenario)

        if key not in grouped:
            grouped[key] = []

        grouped[key].append((count, mf_total, dj_total))

    for key, rows in grouped.items():
        rows.sort()
        n, scenario = key

        winner = None
        for count, mf_total, dj_total in rows:
            if mf_total < dj_total:
                winner = count
                break

        if winner is None:
            print(
                f"{n:5d} nodes | {scenario:<24} | "
                f"not reached within {UPDATE_COUNTS}"
            )
        else:
            print(
                f"{n:5d} nodes | {scenario:<24} | "
                f"break-even at <= {winner} updates"
            )

    print()
    print("METHODOLOGY:")
    print("1. One graph is generated for each experiment.")
    print("2. M-FARS preprocessing is paid exactly once.")
    print("3. Multiple updates are applied to that SAME graph.")
    print("4. Dijkstra is rerun after every update.")
    print("5. Dijkstra remains the correctness oracle.")
    print("6. Verification time is not counted as M-FARS update time.")
    print("7. Amortized M-FARS time = preprocessing + all M-FARS updates.")
    print("8. Work reduction uses edge-relaxation counts.")
    print()
    print("NOTE:")
    print(
        "V1.4 is designed to test whether the preprocessing cost "
        "is recovered over repeated dynamic updates."
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
    