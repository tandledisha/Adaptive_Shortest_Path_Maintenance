"""
M-FARS VERSION 1.1
Multi-Factor Adaptive Recalculation Strategy
Dynamic weighted graph + Dijkstra baseline

V1.1 adds a REAL regional candidate search:
- SAFE_REUSE: no shortest-path repair is needed.
- REGIONAL_REPAIR: search a local region around the changed edge.
- FULL_RECOMPUTATION: run Dijkstra on the whole graph.

IMPORTANT:
For research correctness, every M-FARS answer is verified with full
Dijkstra. The verification time is reported separately and is NOT
counted as M-FARS decision time.

This is an experimental prototype, not a claim of optimal dynamic SSSP.
"""

import heapq
import random
import time


# ============================================================
# 1. GRAPH GENERATION
# ============================================================

def generate_graph(number_of_nodes, number_of_edges, seed=None):
    """Create a connected undirected positive-weight graph."""
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

    # Add random edges.
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


def copy_graph(graph):
    return {u: dict(neighbors) for u, neighbors in graph.items()}


# ============================================================
# 2. DIJKSTRA
# ============================================================

def dijkstra(graph, source, target):
    """Return shortest distance, path, processed nodes, relaxations."""
    infinity = float("inf")
    distance = {node: infinity for node in graph}
    previous = {node: None for node in graph}

    distance[source] = 0
    heap = [(0, source)]

    processed = 0
    relaxations = 0

    while heap:
        current_distance, u = heapq.heappop(heap)

        if current_distance != distance[u]:
            continue

        processed += 1

        if u == target:
            break

        for v, weight in graph[u].items():
            new_distance = current_distance + weight
            relaxations += 1

            if new_distance < distance[v]:
                distance[v] = new_distance
                previous[v] = u
                heapq.heappush(heap, (new_distance, v))

    path = reconstruct_path(previous, source, target)

    return distance[target], path, processed, relaxations


def reconstruct_path(previous, source, target):
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
    return {
        tuple(sorted((path[i], path[i + 1])))
        for i in range(len(path) - 1)
    }


def get_all_edges(graph):
    edges = []

    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.append((u, v))

    return edges


def choose_update_edge(graph, path, scenario):
    """
    Choose an edge according to the experiment scenario.
    """
    all_edges = get_all_edges(graph)
    shortest_edges = list(path_edges(path))

    if scenario == "RANDOM_SMALL":
        candidates = all_edges

    elif scenario == "RANDOM_LARGE":
        candidates = all_edges

    elif scenario == "SHORTEST_PATH_SMALL":
        candidates = shortest_edges if shortest_edges else all_edges

    elif scenario == "SHORTEST_PATH_LARGE":
        candidates = shortest_edges if shortest_edges else all_edges

    else:
        candidates = all_edges

    return random.choice(candidates)


def apply_update(graph, edge, update_type):
    """Apply an increase or decrease and return old/new weights."""
    u, v = edge
    old_weight = graph[u][v]

    if update_type == "increase_small":
        new_weight = old_weight + max(1, int(round(old_weight * 0.05)))

    elif update_type == "increase_large":
        new_weight = old_weight + max(1, int(round(old_weight * 0.50)))

    elif update_type == "decrease_small":
        new_weight = max(1, old_weight - max(1, int(round(old_weight * 0.05))))

    elif update_type == "decrease_large":
        new_weight = max(1, old_weight - max(1, int(round(old_weight * 0.50))))

    else:
        raise ValueError("Unknown update type")

    graph[u][v] = new_weight
    graph[v][u] = new_weight

    return old_weight, new_weight


# ============================================================
# 4. MULTI-FACTOR SCORE
# ============================================================

def calculate_factors(graph, old_path, edge, old_weight, new_weight):
    """
    Calculate three normalized factors.

    M = update magnitude
    C = path criticality
    R = local region estimate
    """
    # Update magnitude.
    if old_weight == 0:
        magnitude = 1.0
    else:
        magnitude = min(
            1.0,
            abs(new_weight - old_weight) / old_weight
        )

    # Criticality: changed edge belongs to current shortest path.
    edge_key = tuple(sorted(edge))
    current_path_edges = path_edges(old_path)
    criticality = 1.0 if edge_key in current_path_edges else 0.0

    # Local region estimate based on degree of the changed endpoints.
    u, v = edge
    total_degree = max(1, sum(len(graph[node]) for node in graph))
    local_degree = len(graph[u]) + len(graph[v])

    region = min(1.0, local_degree / total_degree * len(graph) * 2)

    # Weighted score.
    score = (
        0.30 * magnitude
        + 0.50 * criticality
        + 0.20 * region
    )

    return magnitude, criticality, region, score


# ============================================================
# 5. REGIONAL REPAIR
# ============================================================

def build_region(graph, edge, radius=2):
    """
    Build a local node region around both endpoints.

    radius=2 means:
        endpoint
        -> neighbors
        -> neighbors of neighbors
    """
    u, v = edge

    region = {u, v}
    frontier = {u, v}

    for _ in range(radius):
        next_frontier = set()

        for node in frontier:
            for neighbor in graph[node]:
                if neighbor not in region:
                    region.add(neighbor)
                    next_frontier.add(neighbor)

        frontier = next_frontier

        if not frontier:
            break

    return region


def regional_repair(graph, source, target, changed_edge, radius=2):
    """
    Search the induced local region.

    Boundary nodes are treated as candidate exits. To avoid claiming
    correctness from a restricted graph alone, the caller verifies the
    result against full Dijkstra.

    Returns:
        distance, path, processed_nodes, relaxations, region_size
    """
    region = build_region(graph, changed_edge, radius)

    # If source/target is outside the local region, regional repair
    # cannot safely answer the query.
    if source not in region or target not in region:
        return None, [], 0, 0, len(region)

    infinity = float("inf")
    distance = {node: infinity for node in region}
    previous = {node: None for node in region}

    distance[source] = 0
    heap = [(0, source)]

    processed = 0
    relaxations = 0

    while heap:
        current_distance, u = heapq.heappop(heap)

        if current_distance != distance[u]:
            continue

        processed += 1

        if u == target:
            break

        for v, weight in graph[u].items():
            if v not in region:
                continue

            new_distance = current_distance + weight
            relaxations += 1

            if new_distance < distance[v]:
                distance[v] = new_distance
                previous[v] = u
                heapq.heappush(heap, (new_distance, v))

    if distance[target] == infinity:
        return None, [], processed, relaxations, len(region)

    path = reconstruct_path(previous, source, target)

    return (
        distance[target],
        path,
        processed,
        relaxations,
        len(region)
    )


# ============================================================
# 6. M-FARS V1.1 DECISION
# ============================================================

def m_fars_v1_1(
    graph,
    source,
    target,
    old_path,
    changed_edge,
    old_weight,
    new_weight
):
    """
    Select an adaptive strategy and execute it.

    SAFE_REUSE:
        Edge increased and was not on the old shortest path.

    REGIONAL_REPAIR:
        Changed edge is important, but the local region contains both
        source and target.

    FULL_RECOMPUTATION:
        Conservative fallback.
    """
    magnitude, criticality, region_factor, score = calculate_factors(
        graph,
        old_path,
        changed_edge,
        old_weight,
        new_weight
    )

    # Very important safe rule:
    # Increasing an edge that is NOT on the current shortest path cannot
    # make the current path worse. Therefore it can safely be reused.
    if new_weight > old_weight and criticality == 0.0:
        strategy = "SAFE_REUSE"

        return {
            "strategy": strategy,
            "distance": None,
            "path": list(old_path),
            "processed": 0,
            "relaxations": 0,
            "region_size": 0,
            "score": score,
            "magnitude": magnitude,
            "criticality": criticality,
            "region_factor": region_factor,
        }

    # If the update is potentially important, attempt regional repair.
    regional_distance, regional_path, processed, relaxations, region_size = (
        regional_repair(
            graph,
            source,
            target,
            changed_edge,
            radius=2
        )
    )

    if regional_distance is not None:
        strategy = "REGIONAL_REPAIR"

        return {
            "strategy": strategy,
            "distance": regional_distance,
            "path": regional_path,
            "processed": processed,
            "relaxations": relaxations,
            "region_size": region_size,
            "score": score,
            "magnitude": magnitude,
            "criticality": criticality,
            "region_factor": region_factor,
        }

    # Conservative fallback.
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
        "region_size": region_size,
        "score": score,
        "magnitude": magnitude,
        "criticality": criticality,
        "region_factor": region_factor,
        "internal_dijkstra_time": elapsed,
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

        # Original shortest path.
        old_distance, old_path, _, _ = dijkstra(
            graph,
            source,
            target
        )

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

        # M-FARS timing only.
        start = time.perf_counter()

        mfars = m_fars_v1_1(
            graph,
            source,
            target,
            old_path,
            changed_edge,
            old_weight,
            new_weight
        )

        mfars_time = time.perf_counter() - start

        # Full Dijkstra oracle.
        start = time.perf_counter()

        true_distance, true_path, true_processed, true_relaxations = (
            dijkstra(
                graph,
                source,
                target
            )
        )

        verification_time = time.perf_counter() - start

        # SAFE_REUSE does not calculate a new distance.
        # If reused path is returned, calculate its actual cost.
        if mfars["strategy"] == "SAFE_REUSE":
            mfars_distance = 0

            for i in range(len(old_path) - 1):
                u = old_path[i]
                v = old_path[i + 1]
                mfars_distance += graph[u][v]

            mfars["distance"] = mfars_distance

        # Exact correctness test.
        correct = (
            mfars["distance"] == true_distance
            and mfars["path"] == true_path
        )

        path_changed = old_path != true_path

        results.append({
            "strategy": mfars["strategy"],
            "score": mfars["score"],
            "mfars_time": mfars_time,
            "verification_time": verification_time,
            "mfars_processed": mfars["processed"],
            "mfars_relaxations": mfars["relaxations"],
            "region_size": mfars["region_size"],
            "true_processed": true_processed,
            "true_relaxations": true_relaxations,
            "correct": correct,
            "path_changed": path_changed,
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
    avg_verification = sum(r["verification_time"] for r in results) / count
    avg_mfars_nodes = sum(r["mfars_processed"] for r in results) / count
    avg_mfars_relaxations = (
        sum(r["mfars_relaxations"] for r in results) / count
    )
    avg_region = sum(r["region_size"] for r in results) / count
    avg_true_nodes = sum(r["true_processed"] for r in results) / count
    avg_true_relaxations = (
        sum(r["true_relaxations"] for r in results) / count
    )

    correctness = (
        sum(1 for r in results if r["correct"]) / count * 100
    )

    path_changed = (
        sum(1 for r in results if r["path_changed"]) / count * 100
    )

    strategy_counts = {}

    for result in results:
        strategy = result["strategy"]
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print()
    print("-" * 62)
    print(f"Graph size: {graph_size} nodes")
    print(f"Edges: {edges}")
    print("-" * 62)
    print()
    print(f"Scenario: {scenario}")
    print(f"Update: {update_type}")
    print(f"Experiments: {count}")
    print(f"Average M-FARS score: {avg_score:.4f}")
    print(f"Path changed: {path_changed:.2f}%")
    print(f"Correctness: {correctness:.2f}%")
    print(f"Average M-FARS time: {avg_mfars_time:.8f} seconds")
    print(
        f"Average full-Dijkstra verification time: "
        f"{avg_verification:.8f} seconds"
    )
    print(f"Average M-FARS nodes processed: {avg_mfars_nodes:.2f}")
    print(
        f"Average M-FARS edge relaxations: "
        f"{avg_mfars_relaxations:.2f}"
    )
    print(f"Average regional nodes: {avg_region:.2f}")
    print(f"Average full-Dijkstra nodes: {avg_true_nodes:.2f}")
    print(
        f"Average full-Dijkstra edge relaxations: "
        f"{avg_true_relaxations:.2f}"
    )

    print()
    print("Strategy distribution:")

    for strategy in (
        "SAFE_REUSE",
        "REGIONAL_REPAIR",
        "FULL_RECOMPUTATION"
    ):
        value = strategy_counts.get(strategy, 0)
        percentage = value / count * 100
        print(f"  {strategy:20s}: {value:3d} ({percentage:6.2f}%)")


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 62)
    print("M-FARS VERSION 1.1")
    print("MULTI-FACTOR ADAPTIVE RECALCULATION STRATEGY")
    print("REGIONAL REPAIR EXPERIMENT")
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
        "REGIONAL_REPAIR": 0,
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
                seed=1000 + nodes
            )

            print_results(
                nodes,
                edges,
                scenario,
                update_type,
                results
            )

            total_experiments += len(results)
            total_correct += sum(
                1 for result in results if result["correct"]
            )

            for result in results:
                strategy = result["strategy"]
                total_strategy_counts[strategy] += 1

    print()
    print("=" * 62)
    print("M-FARS V1.1 FINAL SUMMARY")
    print("=" * 62)

    overall_correctness = (
        total_correct / total_experiments * 100
    )

    print(f"Total experiments: {total_experiments}")
    print(f"Overall correctness: {overall_correctness:.2f}%")
    print()
    print("Strategy counts:")

    for strategy, count in total_strategy_counts.items():
        percentage = count / total_experiments * 100
        print(
            f"  {strategy:20s}: {count:3d} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("IMPORTANT:")
    print("Full Dijkstra is run after every experiment only as a")
    print("correctness oracle. Its time is NOT counted as M-FARS time.")
    print()
    print("V1.1 tests whether a local regional repair can reduce")
    print("the amount of work compared with full recomputation.")


if __name__ == "__main__":
    main()