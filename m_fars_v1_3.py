"""
M-FARS VERSION 1.3
FAIR DYNAMIC SHORTEST-PATH BENCHMARK

Purpose
-------
V1.3 does NOT introduce a new heuristic. It fixes the experimental
methodology so that M-FARS can be compared fairly with a target-specific
Dijkstra baseline.

Important methodological choices
--------------------------------
1. Dijkstra stops as soon as the target is finalized.
2. Preprocessing time is measured separately.
3. M-FARS update time is measured separately.
4. An amortized M-FARS time is also reported:
       preprocessing_time / number_of_updates + update_time
5. Correctness compares shortest-path DISTANCE, not the exact path,
   because multiple shortest paths can have the same optimal cost.
6. Work reduction is reported using edge relaxations.
7. Speedup is reported against target-specific Dijkstra.
8. Each experiment performs ONE graph update after preprocessing. This
   keeps the experiment easy to understand and avoids silently assuming
   that distance labels remain valid after an earlier update.

V1.2 strategy retained
----------------------
SAFE_REUSE:
    Increase an edge that is not on the old shortest path.

EXACT_DECREASE:
    For one edge decrease, use old source/target distance labels to
    evaluate the best path that crosses the changed edge.

FULL_RECOMPUTATION:
    Increase of an edge on the old shortest path.

This file is a benchmark version, not the final research algorithm.
"""

import heapq
import random
import time


# ============================================================
# 1. GRAPH GENERATION
# ============================================================

def generate_graph(number_of_nodes, number_of_edges, seed=None):
    """Generate a connected undirected positive-weight graph."""
    if seed is not None:
        random.seed(seed)

    graph = {i: {} for i in range(number_of_nodes)}

    # First create a connected backbone.
    for node in range(1, number_of_nodes):
        parent = random.randrange(node)
        weight = random.randint(1, 20)

        graph[node][parent] = weight
        graph[parent][node] = weight

    current_edges = number_of_nodes - 1
    max_edges = number_of_nodes * (number_of_nodes - 1) // 2
    number_of_edges = min(number_of_edges, max_edges)

    # Then add random edges.
    while current_edges < number_of_edges:
        u = random.randrange(number_of_nodes)
        v = random.randrange(number_of_nodes)

        if u == v or v in graph[u]:
            continue

        weight = random.randint(1, 20)

        graph[u][v] = weight
        graph[v][u] = weight
        current_edges += 1

    return graph


# ============================================================
# 2. TARGET-SPECIFIC DIJKSTRA
# ============================================================

def dijkstra_target(graph, source, target):
    """
    Dijkstra from source to target.

    IMPORTANT:
        Stops when target is extracted from the priority queue.

    Returns:
        distance
        path
        processed_nodes
        edge_relaxations
    """
    infinity = float("inf")

    distances = {node: infinity for node in graph}
    previous = {node: None for node in graph}

    distances[source] = 0
    heap = [(0, source)]

    processed_nodes = 0
    edge_relaxations = 0

    while heap:
        current_distance, u = heapq.heappop(heap)

        if current_distance != distances[u]:
            continue

        processed_nodes += 1

        # Target-specific stopping condition.
        if u == target:
            break

        for v, weight in graph[u].items():
            edge_relaxations += 1

            new_distance = current_distance + weight

            if new_distance < distances[v]:
                distances[v] = new_distance
                previous[v] = u
                heapq.heappush(heap, (new_distance, v))

    path = reconstruct_path(previous, source, target)

    return (
        distances[target],
        path,
        processed_nodes,
        edge_relaxations,
    )


def dijkstra_all(graph, source):
    """
    Full Dijkstra used only for preprocessing distance labels.

    Returns:
        distances
        previous
        processed_nodes
        edge_relaxations
    """
    infinity = float("inf")

    distances = {node: infinity for node in graph}
    previous = {node: None for node in graph}

    distances[source] = 0
    heap = [(0, source)]

    processed_nodes = 0
    edge_relaxations = 0

    while heap:
        current_distance, u = heapq.heappop(heap)

        if current_distance != distances[u]:
            continue

        processed_nodes += 1

        for v, weight in graph[u].items():
            edge_relaxations += 1

            new_distance = current_distance + weight

            if new_distance < distances[v]:
                distances[v] = new_distance
                previous[v] = u
                heapq.heappush(heap, (new_distance, v))

    return (
        distances,
        previous,
        processed_nodes,
        edge_relaxations,
    )


def reconstruct_path(previous, source, target):
    """Reconstruct a path from predecessor information."""
    if source == target:
        return [source]

    if previous[target] is None:
        return []

    path = []
    current = target

    while current is not None:
        path.append(current)

        if current == source:
            break

        current = previous[current]

    path.reverse()

    if not path or path[0] != source:
        return []

    return path


# ============================================================
# 3. EDGE / PATH UTILITIES
# ============================================================

def get_all_edges(graph):
    """Return each undirected edge exactly once."""
    edges = []

    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.append((u, v))

    return edges


def path_edges(path):
    """Return undirected edges in a path."""
    return {
        tuple(sorted((path[i], path[i + 1])))
        for i in range(len(path) - 1)
    }


def choose_update_edge(graph, old_path, scenario):
    """Choose a random or shortest-path edge according to scenario."""
    all_edges = get_all_edges(graph)

    if scenario.startswith("SHORTEST_PATH"):
        candidates = list(path_edges(old_path))

        if not candidates:
            candidates = all_edges
    else:
        candidates = all_edges

    return random.choice(candidates)


def apply_update(graph, edge, update_type):
    """Change one edge weight and return old/new weights."""
    u, v = edge
    old_weight = graph[u][v]

    if update_type == "increase_small":
        delta = max(1, int(round(old_weight * 0.05)))
        new_weight = old_weight + delta

    elif update_type == "increase_large":
        delta = max(1, int(round(old_weight * 0.50)))
        new_weight = old_weight + delta

    elif update_type == "decrease_small":
        delta = max(1, int(round(old_weight * 0.05)))
        new_weight = max(1, old_weight - delta)

    elif update_type == "decrease_large":
        delta = max(1, int(round(old_weight * 0.50)))
        new_weight = max(1, old_weight - delta)

    else:
        raise ValueError("Unknown update type.")

    graph[u][v] = new_weight
    graph[v][u] = new_weight

    return old_weight, new_weight


# ============================================================
# 4. M-FARS FACTORS
# ============================================================

def calculate_factors(old_path, changed_edge, old_weight, new_weight):
    """Calculate update magnitude, criticality and direction."""
    if old_weight == 0:
        magnitude = 1.0
    else:
        magnitude = min(
            1.0,
            abs(new_weight - old_weight) / old_weight
        )

    criticality = (
        1.0
        if tuple(sorted(changed_edge)) in path_edges(old_path)
        else 0.0
    )

    if new_weight > old_weight:
        direction = "INCREASE"
    elif new_weight < old_weight:
        direction = "DECREASE"
    else:
        direction = "UNCHANGED"

    # This score is a reported research feature.
    # It is NOT used to override correctness rules.
    direction_factor = 1.0 if direction != "UNCHANGED" else 0.0

    score = (
        0.35 * magnitude
        + 0.50 * criticality
        + 0.15 * direction_factor
    )

    return magnitude, criticality, direction, score


# ============================================================
# 5. EXACT SINGLE-EDGE DECREASE
# ============================================================

def exact_decrease_distance(
    source,
    target,
    edge,
    new_weight,
    old_source_distances,
    old_target_distances,
    old_distance,
):
    """
    Exact distance after a single undirected edge decrease.

    If the changed edge is (u,v), any newly improved shortest path must
    use that edge. Therefore:

      min(
          old_dist_s[u] + new_w + old_dist_t[v],
          old_dist_s[v] + new_w + old_dist_t[u],
          old_dist_s[target]
      )

    Only the distance is needed for the benchmark correctness test.
    """
    u, v = edge

    candidate_uv = (
        old_source_distances[u]
        + new_weight
        + old_target_distances[v]
    )

    candidate_vu = (
        old_source_distances[v]
        + new_weight
        + old_target_distances[u]
    )

    return min(
        old_distance,
        candidate_uv,
        candidate_vu,
    )


# ============================================================
# 6. M-FARS V1.3 UPDATE
# ============================================================

def m_fars_update(
    graph,
    source,
    target,
    old_path,
    old_distance,
    old_source_distances,
    old_target_distances,
    changed_edge,
    old_weight,
    new_weight,
):
    """
    Perform one M-FARS update.

    Returns a dictionary of update-level metrics.
    """
    magnitude, criticality, direction, score = calculate_factors(
        old_path,
        changed_edge,
        old_weight,
        new_weight,
    )

    # --------------------------------------------------------
    # CASE A: SAFE REUSE
    # --------------------------------------------------------
    if direction == "INCREASE" and criticality == 0.0:
        return {
            "strategy": "SAFE_REUSE",
            "distance": old_distance,
            "processed": 0,
            "relaxations": 0,
            "score": score,
            "magnitude": magnitude,
            "criticality": criticality,
        }

    # --------------------------------------------------------
    # CASE B: EXACT DECREASE
    # --------------------------------------------------------
    if direction == "DECREASE":
        distance = exact_decrease_distance(
            source,
            target,
            changed_edge,
            new_weight,
            old_source_distances,
            old_target_distances,
            old_distance,
        )

        return {
            "strategy": "EXACT_DECREASE",
            "distance": distance,
            "processed": 0,
            "relaxations": 0,
            "score": score,
            "magnitude": magnitude,
            "criticality": criticality,
        }

    # --------------------------------------------------------
    # CASE C: FULL RECOMPUTATION
    # --------------------------------------------------------
    distance, path, processed, relaxations = dijkstra_target(
        graph,
        source,
        target,
    )

    return {
        "strategy": "FULL_RECOMPUTATION",
        "distance": distance,
        "path": path,
        "processed": processed,
        "relaxations": relaxations,
        "score": score,
        "magnitude": magnitude,
        "criticality": criticality,
    }


# ============================================================
# 7. ONE EXPERIMENT
# ============================================================

def run_one_experiment(
    number_of_nodes,
    number_of_edges,
    scenario,
    update_type,
    seed,
):
    """
    Run one independent graph/update experiment.

    The graph is generated once.

    Preprocessing:
        source and target distance labels are computed before the update.

    Update:
        one edge is modified and M-FARS responds.

    Baseline:
        target-specific Dijkstra runs after the update.
    """
    random.seed(seed)

    graph = generate_graph(
        number_of_nodes,
        number_of_edges,
        seed=seed,
    )

    source = 0
    target = number_of_nodes - 1

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------
    start = time.perf_counter()

    source_distances, source_previous, prep_source_nodes, prep_source_relax = (
        dijkstra_all(graph, source)
    )

    target_distances, _, prep_target_nodes, prep_target_relax = (
        dijkstra_all(graph, target)
    )

    preprocessing_time = time.perf_counter() - start

    old_distance = source_distances[target]
    old_path = reconstruct_path(
        source_previous,
        source,
        target,
    )

    # --------------------------------------------------------
    # SELECT AND APPLY UPDATE
    # --------------------------------------------------------
    changed_edge = choose_update_edge(
        graph,
        old_path,
        scenario,
    )

    old_weight, new_weight = apply_update(
        graph,
        changed_edge,
        update_type,
    )

    # --------------------------------------------------------
    # M-FARS UPDATE TIME
    # --------------------------------------------------------
    start = time.perf_counter()

    mfars = m_fars_update(
        graph,
        source,
        target,
        old_path,
        old_distance,
        source_distances,
        target_distances,
        changed_edge,
        old_weight,
        new_weight,
    )

    mfars_update_time = time.perf_counter() - start

    # --------------------------------------------------------
    # BASELINE: TARGET-SPECIFIC DIJKSTRA
    # --------------------------------------------------------
    start = time.perf_counter()

    baseline_distance, baseline_path, baseline_nodes, baseline_relax = (
        dijkstra_target(
            graph,
            source,
            target,
        )
    )

    baseline_time = time.perf_counter() - start

    # --------------------------------------------------------
    # CORRECTNESS
    # --------------------------------------------------------
    #
    # Compare distance, not exact path.
    #
    # Multiple shortest paths can legitimately have the same cost.
    #
    correct = mfars["distance"] == baseline_distance

    # --------------------------------------------------------
    # WORK / SPEEDUP METRICS
    # --------------------------------------------------------
    if baseline_relax > 0:
        work_reduction = (
            1.0
            - mfars["relaxations"] / baseline_relax
        ) * 100.0
    else:
        work_reduction = 0.0

    if mfars_update_time > 0:
        update_speedup = (
            baseline_time / mfars_update_time
        )
    else:
        update_speedup = float("inf")

    # Preprocessing is paid once per graph.
    # Because this benchmark has one update per graph, the amortized
    # cost is equal to preprocessing + update time.
    amortized_total_time = (
        preprocessing_time + mfars_update_time
    )

    if amortized_total_time > 0:
        amortized_speedup = (
            baseline_time / amortized_total_time
        )
    else:
        amortized_speedup = float("inf")

    return {
        "strategy": mfars["strategy"],
        "score": mfars["score"],
        "criticality": mfars["criticality"],
        "magnitude": mfars["magnitude"],
        "correct": correct,

        "preprocessing_time": preprocessing_time,
        "mfars_update_time": mfars_update_time,
        "baseline_time": baseline_time,
        "amortized_total_time": amortized_total_time,

        "mfars_nodes": mfars["processed"],
        "mfars_relaxations": mfars["relaxations"],

        "baseline_nodes": baseline_nodes,
        "baseline_relaxations": baseline_relax,

        "work_reduction": work_reduction,
        "update_speedup": update_speedup,
        "amortized_speedup": amortized_speedup,

        "path_changed": (
            old_path != baseline_path
        ),

        "prep_source_nodes": prep_source_nodes,
        "prep_source_relax": prep_source_relax,
        "prep_target_nodes": prep_target_nodes,
        "prep_target_relax": prep_target_relax,

        "old_weight": old_weight,
        "new_weight": new_weight,
        "changed_edge": changed_edge,
    }


# ============================================================
# 8. AGGREGATION
# ============================================================

def average(results, key):
    """Average one numeric result field."""
    if not results:
        return 0.0

    values = [
        result[key]
        for result in results
        if isinstance(result[key], (int, float))
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def print_results(
    graph_size,
    edges,
    scenario,
    update_type,
    results,
):
    """Print a complete benchmark block."""
    count = len(results)

    correctness = (
        sum(result["correct"] for result in results)
        / count
        * 100.0
    )

    path_changed = (
        sum(result["path_changed"] for result in results)
        / count
        * 100.0
    )

    avg_prep = average(results, "preprocessing_time")
    avg_mfars = average(results, "mfars_update_time")
    avg_baseline = average(results, "baseline_time")
    avg_amortized = average(results, "amortized_total_time")

    avg_mfars_nodes = average(results, "mfars_nodes")
    avg_mfars_relax = average(results, "mfars_relaxations")

    avg_baseline_nodes = average(results, "baseline_nodes")
    avg_baseline_relax = average(results, "baseline_relaxations")

    avg_work_reduction = average(results, "work_reduction")
    avg_update_speedup = average(results, "update_speedup")
    avg_amortized_speedup = average(
        results,
        "amortized_speedup",
    )

    print()
    print("=" * 68)
    print(f"Graph size: {graph_size} nodes")
    print(f"Edges: {edges}")
    print("=" * 68)
    print()
    print(f"Scenario: {scenario}")
    print(f"Update: {update_type}")
    print(f"Experiments: {count}")
    print()
    print(f"Average M-FARS score: {average(results, 'score'):.4f}")
    print(f"Path changed: {path_changed:.2f}%")
    print(f"Correctness: {correctness:.2f}%")
    print()

    print("TIME:")
    print(f"  Preprocessing time:       {avg_prep:.8f} seconds")
    print(f"  M-FARS update time:       {avg_mfars:.8f} seconds")
    print(f"  Dijkstra update time:     {avg_baseline:.8f} seconds")
    print(f"  M-FARS amortized total:   {avg_amortized:.8f} seconds")
    print()

    print("NODES:")
    print(f"  M-FARS nodes:             {avg_mfars_nodes:.2f}")
    print(f"  Dijkstra nodes:           {avg_baseline_nodes:.2f}")
    print()

    print("EDGE RELAXATIONS:")
    print(f"  M-FARS relaxations:       {avg_mfars_relax:.2f}")
    print(f"  Dijkstra relaxations:     {avg_baseline_relax:.2f}")
    print()

    print("EFFICIENCY:")
    print(f"  Work reduction:           {avg_work_reduction:.2f}%")
    print(f"  Update-only speedup:      {avg_update_speedup:.3f}x")
    print(f"  Amortized speedup:        {avg_amortized_speedup:.3f}x")
    print()

    print("STRATEGY DISTRIBUTION:")

    strategy_names = [
        "SAFE_REUSE",
        "EXACT_DECREASE",
        "FULL_RECOMPUTATION",
    ]

    for strategy in strategy_names:
        strategy_count = sum(
            result["strategy"] == strategy
            for result in results
        )

        percentage = strategy_count / count * 100.0

        print(
            f"  {strategy:20s}: "
            f"{strategy_count:3d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

def print_final_summary(all_results):
    """Print global benchmark summary."""
    total = len(all_results)

    correct = sum(
        result["correct"]
        for result in all_results
    )

    correctness = correct / total * 100.0

    avg_prep = average(
        all_results,
        "preprocessing_time",
    )

    avg_mfars = average(
        all_results,
        "mfars_update_time",
    )

    avg_baseline = average(
        all_results,
        "baseline_time",
    )

    avg_amortized = average(
        all_results,
        "amortized_total_time",
    )

    avg_work = average(
        all_results,
        "work_reduction",
    )

    avg_update_speedup = average(
        all_results,
        "update_speedup",
    )

    avg_amortized_speedup = average(
        all_results,
        "amortized_speedup",
    )

    print()
    print("=" * 68)
    print("M-FARS V1.3 FINAL FAIR-BENCHMARK SUMMARY")
    print("=" * 68)
    print()
    print(f"Total experiments:       {total}")
    print(f"Overall correctness:     {correctness:.2f}%")
    print()
    print("AVERAGES:")
    print(f"Preprocessing time:      {avg_prep:.8f} seconds")
    print(f"M-FARS update time:      {avg_mfars:.8f} seconds")
    print(f"Dijkstra update time:    {avg_baseline:.8f} seconds")
    print(f"M-FARS amortized total:  {avg_amortized:.8f} seconds")
    print(f"Work reduction:          {avg_work:.2f}%")
    print(f"Update-only speedup:     {avg_update_speedup:.3f}x")
    print(f"Amortized speedup:       {avg_amortized_speedup:.3f}x")
    print()
    print("STRATEGY COUNTS:")

    for strategy in (
        "SAFE_REUSE",
        "EXACT_DECREASE",
        "FULL_RECOMPUTATION",
    ):
        count = sum(
            result["strategy"] == strategy
            for result in all_results
        )

        percentage = count / total * 100.0

        print(
            f"  {strategy:20s}: "
            f"{count:3d} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("METHODOLOGY:")
    print("1. Dijkstra is target-specific and stops at the target.")
    print("2. Preprocessing is measured separately.")
    print("3. M-FARS update computation is measured separately.")
    print("4. Amortized time includes preprocessing.")
    print("5. Work reduction uses edge-relaxation counts.")
    print("6. Correctness compares shortest-path distance.")
    print()
    print("NOTE:")
    print("This benchmark uses one update per independently")
    print("generated graph. Therefore amortized time is a")
    print("conservative one-update accounting of preprocessing.")
    print("A future multi-update experiment can amortize the")
    print("preprocessing cost across many updates.")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    print("=" * 68)
    print("M-FARS VERSION 1.3")
    print("FAIR DYNAMIC SHORTEST-PATH BENCHMARK")
    print("=" * 68)

    graph_sizes = [
        (100, 300),
        (500, 1500),
        (1000, 3000),
        (5000, 15000),
    ]

    scenarios = [
        ("RANDOM_SMALL", "increase_small"),
        ("RANDOM_LARGE", "increase_large"),
        ("SHORTEST_PATH_SMALL", "increase_small"),
        ("SHORTEST_PATH_LARGE", "increase_large"),
        ("RANDOM_DECREASE_SMALL", "decrease_small"),
        ("RANDOM_DECREASE_LARGE", "decrease_large"),
    ]

    experiments_per_case = 10

    all_results = []

    print()
    print("Starting fair experiments...")
    print(
        f"Graphs: {len(graph_sizes)} | "
        f"Scenarios: {len(scenarios)} | "
        f"Experiments/case: {experiments_per_case}"
    )
    print(
        f"Total experiments: "
        f"{len(graph_sizes) * len(scenarios) * experiments_per_case}"
    )

    for nodes, edges in graph_sizes:
        for scenario, update_type in scenarios:

            results = []

            for experiment in range(experiments_per_case):
                seed = (
                    100000
                    + nodes * 100
                    + experiment
                )

                result = run_one_experiment(
                    nodes,
                    edges,
                    scenario,
                    update_type,
                    seed,
                )

                results.append(result)
                all_results.append(result)

            print_results(
                nodes,
                edges,
                scenario,
                update_type,
                results,
            )

    print_final_summary(all_results)


if __name__ == "__main__":
    main()