"""
M-FARS VERSION 1.2
CERTIFIED MULTI-FACTOR ADAPTIVE RECALCULATION STRATEGY

Purpose:
    Fix the correctness problem observed in M-FARS V1.1.

V1.1 problem:
    A radius-limited regional Dijkstra can return a locally shortest path
    that is NOT globally shortest. The V1.1 experiment showed 97.50%
    correctness, so that strategy must not be treated as a final algorithm.

V1.2 uses only mathematically safe shortcuts:

1. SAFE_REUSE
   If an edge weight INCREASES and the changed edge was not on the old
   shortest path, the old shortest path remains valid.

2. EXACT_DECREASE
   If an edge weight DECREASES, we compute the best path that uses the
   changed edge using OLD source/target distance labels. This is exact
   for a single-edge decrease.

3. FULL_RECOMPUTATION
   If an edge on the current shortest path INCREASES, we conservatively
   run full Dijkstra.

This version prioritizes correctness first. The next research step is
to improve the FULL_RECOMPUTATION case with a formally safe dynamic
shortest-path repair method.
"""

import heapq
import random
import time


# ============================================================
# 1. GRAPH GENERATION
# ============================================================

def generate_graph(number_of_nodes, number_of_edges, seed=None):
    """Create a connected undirected graph with positive weights."""
    if seed is not None:
        random.seed(seed)

    graph = {i: {} for i in range(number_of_nodes)}

    # Connected backbone.
    for node in range(1, number_of_nodes):
        parent = random.randrange(node)
        weight = random.randint(1, 20)

        graph[node][parent] = weight
        graph[parent][node] = weight

    current_edges = number_of_nodes - 1
    max_edges = number_of_nodes * (number_of_nodes - 1) // 2
    number_of_edges = min(number_of_edges, max_edges)

    # Extra random edges.
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
# 2. DIJKSTRA
# ============================================================

def dijkstra_all(graph, source):
    """
    Dijkstra from source to every node.

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

    return distances, previous, processed_nodes, edge_relaxations


def dijkstra(graph, source, target):
    """Dijkstra from source to target."""
    distances, previous, processed, relaxations = dijkstra_all(
        graph, source
    )

    path = reconstruct_path(previous, source, target)

    return (
        distances[target],
        path,
        processed,
        relaxations,
    )


def reconstruct_path(previous, source, target):
    """Reconstruct a source-to-target path."""
    if target != source and previous[target] is None:
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
# 3. PATH / EDGE UTILITIES
# ============================================================

def path_edges(path):
    """Return undirected edges contained in a path."""
    return {
        tuple(sorted((path[i], path[i + 1])))
        for i in range(len(path) - 1)
    }


def get_all_edges(graph):
    """Return each undirected edge once."""
    edges = []

    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.append((u, v))

    return edges


def choose_update_edge(graph, path, scenario):
    """Select an edge according to the experiment scenario."""
    all_edges = get_all_edges(graph)
    shortest_edges = list(path_edges(path))

    if scenario.startswith("SHORTEST_PATH"):
        candidates = shortest_edges if shortest_edges else all_edges
    else:
        candidates = all_edges

    return random.choice(candidates)


def apply_update(graph, edge, update_type):
    """Apply one edge-weight update."""
    u, v = edge
    old_weight = graph[u][v]

    if update_type == "increase_small":
        new_weight = old_weight + max(
            1, int(round(old_weight * 0.05))
        )

    elif update_type == "increase_large":
        new_weight = old_weight + max(
            1, int(round(old_weight * 0.50))
        )

    elif update_type == "decrease_small":
        new_weight = max(
            1, old_weight - max(1, int(round(old_weight * 0.05)))
        )

    elif update_type == "decrease_large":
        new_weight = max(
            1, old_weight - max(1, int(round(old_weight * 0.50)))
        )

    else:
        raise ValueError("Unknown update type.")

    graph[u][v] = new_weight
    graph[v][u] = new_weight

    return old_weight, new_weight


# ============================================================
# 4. MULTI-FACTOR MODEL
# ============================================================

def calculate_factors(
    graph,
    old_path,
    edge,
    old_weight,
    new_weight
):
    """
    Calculate the three M-FARS factors.

    M = update magnitude
    C = path criticality
    D = update direction

    The factors are used for reporting and adaptive classification.
    Correctness is enforced by the certified rules in m_fars_v1_2().
    """
    if old_weight == 0:
        magnitude = 1.0
    else:
        magnitude = min(
            1.0,
            abs(new_weight - old_weight) / old_weight
        )

    edge_key = tuple(sorted(edge))

    criticality = (
        1.0
        if edge_key in path_edges(old_path)
        else 0.0
    )

    if new_weight > old_weight:
        direction = 1.0
    elif new_weight < old_weight:
        direction = -1.0
    else:
        direction = 0.0

    # Score is retained as a research feature.
    # It does NOT override the correctness rules.
    score = (
        0.35 * magnitude
        + 0.50 * criticality
        + 0.15 * (1.0 if direction != 0 else 0.0)
    )

    return magnitude, criticality, direction, score


# ============================================================
# 5. EXACT SINGLE-EDGE DECREASE
# ============================================================

def exact_decrease_candidate(
    source,
    target,
    edge,
    new_weight,
    old_source_distances,
    old_target_distances,
    old_target_distance,
    old_source_previous,
    old_target_previous
):
    """
    Compute the exact shortest-path candidate after ONE edge decrease.

    For a single changed undirected edge (u,v), every newly improved
    shortest path must use that changed edge.

    Therefore the new optimum is:

      min(
          old_dist(source,u) + new_w + old_dist(v,target),
          old_dist(source,v) + new_w + old_dist(u,target),
          old_dist(source,target)
      )

    No full post-update Dijkstra is needed for this case.

    The old distance labels are computed BEFORE the update.
    """
    u, v = edge
    infinity = float("inf")

    candidate_a = (
        old_source_distances[u]
        + new_weight
        + old_target_distances[v]
    )

    candidate_b = (
        old_source_distances[v]
        + new_weight
        + old_target_distances[u]
    )

    best_distance = min(
        old_target_distance,
        candidate_a,
        candidate_b
    )

    if best_distance == old_target_distance:
        # Old path remains optimal.
        return old_target_distance, reconstruct_path(
            old_source_previous,
            source,
            target
        )

    # Build the path through the changed edge.
    if candidate_a <= candidate_b:
        left = reconstruct_path(
            old_source_previous,
            source,
            u
        )
        right = reconstruct_path(
            old_target_previous,
            target,
            v
        )

        if not left or not right:
            return infinity, []

        # old_target_previous reconstructs target -> v.
        right = list(reversed(right))

        path = left + [v] + right[1:]

    else:
        left = reconstruct_path(
            old_source_previous,
            source,
            v
        )
        right = reconstruct_path(
            old_target_previous,
            target,
            u
        )

        if not left or not right:
            return infinity, []

        right = list(reversed(right))

        path = left + [u] + right[1:]

    return best_distance, path


# ============================================================
# 6. CERTIFIED M-FARS V1.2
# ============================================================

def m_fars_v1_2(
    graph,
    source,
    target,
    old_path,
    old_distance,
    old_source_distances,
    old_target_distances,
    old_source_previous,
    old_target_previous,
    changed_edge,
    old_weight,
    new_weight,
):
    """Execute the V1.2 certified adaptive strategy."""

    magnitude, criticality, direction, score = calculate_factors(
        graph,
        old_path,
        changed_edge,
        old_weight,
        new_weight
    )

    # --------------------------------------------------------
    # CASE 1: edge increase NOT on shortest path
    # --------------------------------------------------------
    #
    # Increasing an edge that is not on the current shortest path
    # cannot make that path more expensive.
    #
    # Therefore the old shortest path remains globally optimal.
    #
    if direction > 0 and criticality == 0.0:
        return {
            "strategy": "SAFE_REUSE",
            "distance": old_distance,
            "path": list(old_path),
            "processed": 0,
            "relaxations": 0,
            "score": score,
            "magnitude": magnitude,
            "criticality": criticality,
        }

    # --------------------------------------------------------
    # CASE 2: edge decrease
    # --------------------------------------------------------
    #
    # Exact single-edge decrease theorem.
    #
    if direction < 0:
        start = time.perf_counter()

        distance, path = exact_decrease_candidate(
            source,
            target,
            changed_edge,
            new_weight,
            old_source_distances,
            old_target_distances,
            old_distance,
            old_source_previous,
            old_target_previous,
        )

        elapsed = time.perf_counter() - start

        return {
            "strategy": "EXACT_DECREASE",
            "distance": distance,
            "path": path,
            "processed": 0,
            "relaxations": 0,
            "score": score,
            "magnitude": magnitude,
            "criticality": criticality,
            "internal_time": elapsed,
        }

    # --------------------------------------------------------
    # CASE 3: edge increase ON shortest path
    # --------------------------------------------------------
    #
    # This is the difficult dynamic shortest-path case.
    # V1.2 deliberately uses a conservative full recomputation.
    #
    start = time.perf_counter()

    distance, path, processed, relaxations = dijkstra(
        graph,
        source,
        target
    )

    elapsed = time.perf_counter() - start

    return {
        "strategy": "FULL_RECOMPUTATION",
        "distance": distance,
        "path": path,
        "processed": processed,
        "relaxations": relaxations,
        "score": score,
        "magnitude": magnitude,
        "criticality": criticality,
        "internal_time": elapsed,
    }


# ============================================================
# 7. EXPERIMENT
# ============================================================

def run_experiment(
    number_of_nodes,
    number_of_edges,
    scenario,
    update_type,
    experiments=10,
    seed=42
):
    results = []

    for experiment in range(experiments):
        random.seed(seed + experiment)

        graph = generate_graph(
            number_of_nodes,
            number_of_edges,
            seed=seed + experiment
        )

        source = 0
        target = number_of_nodes - 1

        # ----------------------------------------------------
        # OLD GRAPH LABELS
        # ----------------------------------------------------
        #
        # These are computed BEFORE applying the update.
        # They are required by the exact decrease formula.
        #
        old_source_distances, old_source_previous, _, _ = (
            dijkstra_all(graph, source)
        )

        old_target_distances, old_target_previous, _, _ = (
            dijkstra_all(graph, target)
        )

        old_distance = old_source_distances[target]
        old_path = reconstruct_path(
            old_source_previous,
            source,
            target
        )

        # Select and update one edge.
        changed_edge = choose_update_edge(
            graph,
            old_path,
            scenario
        )

        old_weight, new_weight = apply_update(
            graph,
            changed_edge,
            update_type
        )

        # ----------------------------------------------------
        # M-FARS TIMING
        # ----------------------------------------------------
        start = time.perf_counter()

        mfars = m_fars_v1_2(
            graph,
            source,
            target,
            old_path,
            old_distance,
            old_source_distances,
            old_target_distances,
            old_source_previous,
            old_target_previous,
            changed_edge,
            old_weight,
            new_weight,
        )

        mfars_time = time.perf_counter() - start

        # ----------------------------------------------------
        # VERIFICATION ORACLE
        # ----------------------------------------------------
        #
        # Full Dijkstra is intentionally executed separately.
        # It is NOT counted in M-FARS time.
        #
        start = time.perf_counter()

        true_distance, true_path, true_processed, true_relaxations = (
            dijkstra(
                graph,
                source,
                target
            )
        )

        verification_time = time.perf_counter() - start

        correct = (
            mfars["distance"] == true_distance
            and mfars["path"] == true_path
        )

        results.append({
            "strategy": mfars["strategy"],
            "score": mfars["score"],
            "mfars_time": mfars_time,
            "verification_time": verification_time,
            "mfars_processed": mfars["processed"],
            "mfars_relaxations": mfars["relaxations"],
            "true_processed": true_processed,
            "true_relaxations": true_relaxations,
            "correct": correct,
            "path_changed": old_path != true_path,
        })

    return results


# ============================================================
# 8. REPORTING
# ============================================================

def print_results(
    graph_size,
    edges,
    scenario,
    update_type,
    results
):
    count = len(results)

    avg_score = sum(r["score"] for r in results) / count
    avg_mfars_time = sum(r["mfars_time"] for r in results) / count
    avg_verification = (
        sum(r["verification_time"] for r in results) / count
    )

    avg_mfars_nodes = (
        sum(r["mfars_processed"] for r in results) / count
    )

    avg_mfars_relaxations = (
        sum(r["mfars_relaxations"] for r in results) / count
    )

    avg_true_nodes = (
        sum(r["true_processed"] for r in results) / count
    )

    avg_true_relaxations = (
        sum(r["true_relaxations"] for r in results) / count
    )

    correctness = (
        sum(1 for r in results if r["correct"])
        / count
        * 100
    )

    path_changed = (
        sum(1 for r in results if r["path_changed"])
        / count
        * 100
    )

    strategy_counts = {}

    for result in results:
        strategy = result["strategy"]
        strategy_counts[strategy] = (
            strategy_counts.get(strategy, 0) + 1
        )

    print()
    print("=" * 62)
    print(f"Graph size: {graph_size} nodes")
    print(f"Edges: {edges}")
    print("=" * 62)
    print()
    print(f"Scenario: {scenario}")
    print(f"Update: {update_type}")
    print(f"Experiments: {count}")
    print(f"Average M-FARS score: {avg_score:.4f}")
    print(f"Path changed: {path_changed:.2f}%")
    print(f"Correctness: {correctness:.2f}%")
    print(f"Average M-FARS time: {avg_mfars_time:.8f} seconds")
    print(
        "Average full-Dijkstra verification time: "
        f"{avg_verification:.8f} seconds"
    )
    print(
        f"Average M-FARS nodes processed: "
        f"{avg_mfars_nodes:.2f}"
    )
    print(
        f"Average M-FARS edge relaxations: "
        f"{avg_mfars_relaxations:.2f}"
    )
    print(
        f"Average full-Dijkstra nodes: "
        f"{avg_true_nodes:.2f}"
    )
    print(
        f"Average full-Dijkstra edge relaxations: "
        f"{avg_true_relaxations:.2f}"
    )

    print()
    print("Strategy distribution:")

    for strategy in (
        "SAFE_REUSE",
        "EXACT_DECREASE",
        "FULL_RECOMPUTATION",
    ):
        value = strategy_counts.get(strategy, 0)
        percentage = value / count * 100

        print(
            f"  {strategy:20s}: "
            f"{value:3d} ({percentage:6.2f}%)"
        )


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 62)
    print("M-FARS VERSION 1.2")
    print("CERTIFIED MULTI-FACTOR ADAPTIVE RECALCULATION")
    print("CORRECTNESS-FIRST EXPERIMENT")
    print("=" * 62)

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

    total_experiments = 0
    total_correct = 0

    total_strategy_counts = {
        "SAFE_REUSE": 0,
        "EXACT_DECREASE": 0,
        "FULL_RECOMPUTATION": 0,
    }

    for nodes, edges in graph_sizes:

        for scenario, update_type in scenarios:

            results = run_experiment(
                nodes,
                edges,
                scenario,
                update_type,
                experiments=10,
                seed=1000 + nodes,
            )

            print_results(
                nodes,
                edges,
                scenario,
                update_type,
                results,
            )

            total_experiments += len(results)

            total_correct += sum(
                1 for result in results
                if result["correct"]
            )

            for result in results:
                strategy = result["strategy"]
                total_strategy_counts[strategy] += 1

    print()
    print("=" * 62)
    print("M-FARS V1.2 FINAL SUMMARY")
    print("=" * 62)

    overall_correctness = (
        total_correct
        / total_experiments
        * 100
    )

    print(f"Total experiments: {total_experiments}")
    print(
        f"Overall correctness: "
        f"{overall_correctness:.2f}%"
    )

    print()
    print("Strategy counts:")

    for strategy, count in total_strategy_counts.items():
        percentage = count / total_experiments * 100

        print(
            f"  {strategy:20s}: "
            f"{count:3d} ({percentage:6.2f}%)"
        )

    print()
    print("INTERPRETATION:")
    print("V1.1 achieved 97.50% correctness because its")
    print("radius-limited regional search was not globally safe.")
    print()
    print("V1.2 removes that unsafe shortcut.")
    print("It uses certified reuse for off-path increases,")
    print("an exact formula for single-edge decreases, and")
    print("full Dijkstra for the difficult on-path increase case.")
    print()
    print("Full Dijkstra is used as a verification oracle after")
    print("each experiment. Verification time is NOT counted")
    print("as M-FARS runtime.")


if __name__ == "__main__":
    main()