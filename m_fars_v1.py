# ============================================================
# M-FARS PROJECT
# VERSION 1.0: MULTI-FACTOR ADAPTIVE RECALCULATION STRATEGY
# ============================================================
#
# This is the FIRST prototype of M-FARS.
#
# IMPORTANT:
# - This version is designed for experimentation and learning.
# - It does NOT claim that the weights/thresholds are optimal.
# - A full Dijkstra run is used as a verification oracle.
#
# M-FARS V1 uses three factors:
#
#   M = update magnitude
#   C = path criticality
#   R = affected-region estimate
#
# Score:
#
#   S = 0.3*M + 0.5*C + 0.2*R
#
# Decision:
#
#   S < 0.30  -> SAFE_REUSE
#   0.30-0.60 -> REGIONAL_CANDIDATE
#   >= 0.60   -> FULL_RECOMPUTATION
#
# For correctness, V1 only skips full Dijkstra when the changed
# edge is NOT on the current shortest path AND its weight increases.
# In all other cases it falls back to full Dijkstra.
#
# This lets us measure the idea without sacrificing correctness.
# ============================================================

import random
import heapq
import time


# ============================================================
# 1. GRAPH GENERATION
# ============================================================

def generate_graph(number_of_nodes, number_of_edges):
    """
    Generate a connected undirected weighted graph.

    A chain is created first so every node is reachable.
    Additional random edges are then added.
    """

    maximum_edges = (
        number_of_nodes * (number_of_nodes - 1) // 2
    )

    if number_of_edges < number_of_nodes - 1:
        raise ValueError(
            "number_of_edges must be at least number_of_nodes - 1"
        )

    if number_of_edges > maximum_edges:
        raise ValueError(
            "Too many edges for this number of nodes"
        )

    graph = {
        node: {}
        for node in range(number_of_nodes)
    }

    edges = set()

    # Guarantee connectivity.
    for node in range(number_of_nodes - 1):

        u = node
        v = node + 1
        weight = random.randint(1, 100)

        graph[u][v] = weight
        graph[v][u] = weight

        edges.add((u, v))

    # Add remaining random edges.
    while len(edges) < number_of_edges:

        u = random.randint(0, number_of_nodes - 1)
        v = random.randint(0, number_of_nodes - 1)

        if u == v:
            continue

        edge = tuple(sorted((u, v)))

        if edge in edges:
            continue

        weight = random.randint(1, 100)

        graph[u][v] = weight
        graph[v][u] = weight

        edges.add(edge)

    return graph


# ============================================================
# 2. DIJKSTRA
# ============================================================

def dijkstra(graph, source, destination):
    """
    Return:
        distance
        path
        nodes_processed
        edge_relaxations
    """

    distances = {
        node: float("inf")
        for node in graph
    }

    previous = {
        node: None
        for node in graph
    }

    distances[source] = 0

    priority_queue = [(0, source)]

    visited = set()

    nodes_processed = 0
    edge_relaxations = 0

    while priority_queue:

        current_distance, current_node = heapq.heappop(
            priority_queue
        )

        if current_node in visited:
            continue

        visited.add(current_node)
        nodes_processed += 1

        if current_node == destination:
            break

        for neighbor, weight in graph[current_node].items():

            edge_relaxations += 1

            new_distance = (
                current_distance + weight
            )

            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance
                previous[neighbor] = current_node

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbor)
                )

    if distances[destination] == float("inf"):

        return (
            float("inf"),
            [],
            nodes_processed,
            edge_relaxations
        )

    # Reconstruct path.
    path = []

    current = destination

    while current is not None:

        path.append(current)
        current = previous[current]

    path.reverse()

    return (
        distances[destination],
        path,
        nodes_processed,
        edge_relaxations
    )


# ============================================================
# 3. CHECK WHETHER AN EDGE IS ON A PATH
# ============================================================

def edge_is_on_path(u, v, path):
    """
    Return True if edge (u,v) occurs in the supplied path.
    """

    target = tuple(sorted((u, v)))

    for i in range(len(path) - 1):

        path_edge = tuple(
            sorted((path[i], path[i + 1]))
        )

        if path_edge == target:
            return True

    return False


# ============================================================
# 4. CALCULATE UPDATE MAGNITUDE
# ============================================================

def calculate_update_magnitude(old_weight, new_weight):
    """
    M = |new-old| / old

    Example:
        old = 20
        new = 30

        M = 10 / 20 = 0.5
    """

    if old_weight == 0:
        return 1.0

    magnitude = (
        abs(new_weight - old_weight)
        / old_weight
    )

    # Cap at 1 for the V1 score.
    return min(magnitude, 1.0)


# ============================================================
# 5. CALCULATE PATH CRITICALITY
# ============================================================

def calculate_path_criticality(u, v, current_path):
    """
    V1 definition:

        C = 1 if updated edge is on current shortest path
        C = 0 otherwise
    """

    if edge_is_on_path(u, v, current_path):
        return 1.0

    return 0.0


# ============================================================
# 6. CALCULATE AFFECTED REGION ESTIMATE
# ============================================================

def calculate_region_estimate(graph, u, v):
    """
    Simple V1 affected-region estimate.

    We use the number of unique nodes in the 1-hop neighborhood
    of the changed edge divided by the total number of nodes.

    This is intentionally a simple prototype definition.

    It is NOT yet the final theoretical definition of the
    affected region for the research paper.
    """

    affected_nodes = {
        u,
        v
    }

    # Add direct neighbors of u.
    for neighbor in graph[u]:
        affected_nodes.add(neighbor)

    # Add direct neighbors of v.
    for neighbor in graph[v]:
        affected_nodes.add(neighbor)

    region_ratio = (
        len(affected_nodes)
        / len(graph)
    )

    return min(region_ratio, 1.0)


# ============================================================
# 7. CALCULATE M-FARS SCORE
# ============================================================

def calculate_m_fars_score(
    magnitude,
    criticality,
    region_ratio
):
    """
    V1 weighted score:

        S = 0.3*M + 0.5*C + 0.2*R
    """

    score = (
        0.3 * magnitude
        + 0.5 * criticality
        + 0.2 * region_ratio
    )

    return score


# ============================================================
# 8. ADAPTIVE DECISION
# ============================================================

def choose_recalculation_strategy(
    score,
    criticality,
    weight_increased
):
    """
    Convert the M-FARS score into an action.

    IMPORTANT CORRECTNESS RULE:

    If an edge is NOT on the current shortest path and its
    weight increases, the current shortest path cannot become
    more expensive because of that edge.

    Therefore V1 can safely reuse the current path in that case.

    If the edge is on the current shortest path, V1 uses full
    Dijkstra. This keeps the prototype exact.

    The REGIONAL_CANDIDATE state is recorded for future work,
    but V1 does not perform an approximate regional repair yet.
    """

    if (
        criticality == 0.0
        and weight_increased
    ):
        return "SAFE_REUSE"

    if score < 0.30:
        return "LOW_SCORE_BUT_RECOMPUTE"

    if score < 0.60:
        return "REGIONAL_CANDIDATE"

    return "FULL_RECOMPUTATION"


# ============================================================
# 9. CHANGE AN EDGE
# ============================================================

def change_edge_weight(
    graph,
    u,
    v,
    percentage_change
):
    """
    Increase an edge weight by the requested percentage.
    """

    old_weight = graph[u][v]

    change = max(
        1,
        int(old_weight * percentage_change)
    )

    new_weight = old_weight + change

    graph[u][v] = new_weight
    graph[v][u] = new_weight

    return old_weight, new_weight


# ============================================================
# 10. SELECT RANDOM EDGE
# ============================================================

def choose_random_edge(graph):

    edges = []

    for u in graph:

        for v in graph[u]:

            if u < v:
                edges.append((u, v))

    if not edges:
        return None

    return random.choice(edges)


# ============================================================
# 11. SELECT SHORTEST-PATH EDGE
# ============================================================

def choose_shortest_path_edge(path):

    if len(path) < 2:
        return None

    path_edges = []

    for i in range(len(path) - 1):

        path_edges.append(
            (path[i], path[i + 1])
        )

    return random.choice(path_edges)


# ============================================================
# 12. RUN ONE M-FARS EXPERIMENT
# ============================================================

def run_m_fars_experiment(
    number_of_nodes,
    number_of_edges,
    scenario,
    percentage_change
):

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    graph = generate_graph(
        number_of_nodes,
        number_of_edges
    )

    source = 0
    destination = number_of_nodes - 1

    # --------------------------------------------------------
    # Initial shortest path
    # --------------------------------------------------------

    (
        initial_distance,
        initial_path,
        initial_nodes,
        initial_relaxations
    ) = dijkstra(
        graph,
        source,
        destination
    )

    if not initial_path:
        return None

    # --------------------------------------------------------
    # Select update edge
    # --------------------------------------------------------

    if scenario.startswith("RANDOM"):

        selected_edge = choose_random_edge(graph)

    elif scenario.startswith("SHORTEST_PATH"):

        selected_edge = choose_shortest_path_edge(
            initial_path
        )

    else:

        raise ValueError(
            "Unknown scenario: " + scenario
        )

    if selected_edge is None:
        return None

    u, v = selected_edge

    # --------------------------------------------------------
    # Change edge
    # --------------------------------------------------------

    old_weight, new_weight = change_edge_weight(
        graph,
        u,
        v,
        percentage_change
    )

    # --------------------------------------------------------
    # Calculate the three M-FARS factors
    # --------------------------------------------------------

    magnitude = calculate_update_magnitude(
        old_weight,
        new_weight
    )

    criticality = calculate_path_criticality(
        u,
        v,
        initial_path
    )

    region_ratio = calculate_region_estimate(
        graph,
        u,
        v
    )

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    score = calculate_m_fars_score(
        magnitude,
        criticality,
        region_ratio
    )

    # --------------------------------------------------------
    # Choose strategy
    # --------------------------------------------------------

    strategy = choose_recalculation_strategy(
        score,
        criticality,
        new_weight > old_weight
    )

    # --------------------------------------------------------
    # M-FARS execution
    # --------------------------------------------------------

    start_time = time.perf_counter()

    if strategy == "SAFE_REUSE":

        # The current path remains valid for a positive
        # increase to an edge that was not on the current
        # shortest path.

        m_fars_distance = initial_distance
        m_fars_path = initial_path

        m_fars_nodes = 0
        m_fars_relaxations = 0

    else:

        (
            m_fars_distance,
            m_fars_path,
            m_fars_nodes,
            m_fars_relaxations
        ) = dijkstra(
            graph,
            source,
            destination
        )

    m_fars_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # VERIFICATION ORACLE
    #
    # Always run full Dijkstra after M-FARS so that we can
    # determine whether the M-FARS answer is correct.
    # --------------------------------------------------------

    verification_start = time.perf_counter()

    (
        true_distance,
        true_path,
        true_nodes,
        true_relaxations
    ) = dijkstra(
        graph,
        source,
        destination
    )

    verification_time = (
        time.perf_counter() - verification_start
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    correct_distance = (
        m_fars_distance == true_distance
    )

    correct_path_cost = (
        m_fars_distance == true_distance
    )

    # --------------------------------------------------------
    # Return all information
    # --------------------------------------------------------

    return {
        "nodes": number_of_nodes,
        "edges": number_of_edges,
        "scenario": scenario,

        "old_weight": old_weight,
        "new_weight": new_weight,

        "magnitude": magnitude,
        "criticality": criticality,
        "region_ratio": region_ratio,
        "score": score,

        "strategy": strategy,

        "initial_distance": initial_distance,
        "true_distance": true_distance,

        "m_fars_path": m_fars_path,
        "true_path": true_path,

        "path_changed": (
            initial_path != true_path
        ),

        "correct": correct_distance
        and correct_path_cost,

        "m_fars_time": m_fars_time,
        "verification_time": verification_time,

        "m_fars_nodes": m_fars_nodes,
        "m_fars_relaxations": m_fars_relaxations,

        "true_nodes": true_nodes,
        "true_relaxations": true_relaxations
    }


# ============================================================
# 13. PRINT ONE EXPERIMENT
# ============================================================

def print_single_result(result):

    print()
    print("-" * 70)

    print(
        "Scenario:",
        result["scenario"]
    )

    print(
        "Old weight:",
        result["old_weight"]
    )

    print(
        "New weight:",
        result["new_weight"]
    )

    print(
        "Update magnitude:",
        f'{result["magnitude"]:.4f}'
    )

    print(
        "Path criticality:",
        f'{result["criticality"]:.2f}'
    )

    print(
        "Affected-region estimate:",
        f'{result["region_ratio"]:.4f}'
    )

    print(
        "M-FARS score:",
        f'{result["score"]:.4f}'
    )

    print(
        "Strategy:",
        result["strategy"]
    )

    print(
        "True distance:",
        result["true_distance"]
    )

    print(
        "M-FARS distance:",
        result["m_fars_distance"]
        if "m_fars_distance" in result
        else result["true_distance"]
    )

    print(
        "Correct:",
        result["correct"]
    )

    print(
        "M-FARS time:",
        f'{result["m_fars_time"]:.8f}',
        "seconds"
    )

    print(
        "M-FARS nodes processed:",
        result["m_fars_nodes"]
    )

    print(
        "M-FARS edge relaxations:",
        result["m_fars_relaxations"]
    )


# ============================================================
# 14. BENCHMARK M-FARS V1
# ============================================================

def run_benchmark():

    print("=" * 70)
    print("M-FARS VERSION 1.0")
    print("MULTI-FACTOR ADAPTIVE RECALCULATION STRATEGY")
    print("=" * 70)

    graph_sizes = [
        100,
        500,
        1000,
        5000
    ]

    scenarios = [
        ("RANDOM_SMALL", 0.05),
        ("RANDOM_LARGE", 0.50),
        ("SHORTEST_PATH_SMALL", 0.05),
        ("SHORTEST_PATH_LARGE", 0.50)
    ]

    experiments_per_condition = 10

    total_experiments = 0
    total_correct = 0

    strategy_counts = {}

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    for number_of_nodes in graph_sizes:

        number_of_edges = number_of_nodes * 3

        print()
        print("=" * 70)
        print(
            "Graph size:",
            number_of_nodes,
            "nodes"
        )
        print(
            "Edges:",
            number_of_edges
        )
        print("=" * 70)

        for scenario, percentage_change in scenarios:

            results = []

            for experiment in range(
                experiments_per_condition
            ):

                random.seed(
                    500000
                    + number_of_nodes * 100
                    + experiment
                )

                result = run_m_fars_experiment(
                    number_of_nodes,
                    number_of_edges,
                    scenario,
                    percentage_change
                )

                if result is not None:
                    results.append(result)

            # ------------------------------------------------
            # Aggregate results
            # ------------------------------------------------

            if not results:
                continue

            correct_count = sum(
                1
                for result in results
                if result["correct"]
            )

            total_correct += correct_count
            total_experiments += len(results)

            for result in results:

                strategy = result["strategy"]

                strategy_counts[strategy] = (
                    strategy_counts.get(strategy, 0) + 1
                )

            average_score = (
                sum(
                    result["score"]
                    for result in results
                )
                / len(results)
            )

            average_m_fars_time = (
                sum(
                    result["m_fars_time"]
                    for result in results
                )
                / len(results)
            )

            average_true_time = (
                sum(
                    result["verification_time"]
                    for result in results
                )
                / len(results)
            )

            average_m_fars_nodes = (
                sum(
                    result["m_fars_nodes"]
                    for result in results
                )
                / len(results)
            )

            average_true_nodes = (
                sum(
                    result["true_nodes"]
                    for result in results
                )
                / len(results)
            )

            path_change_percentage = (
                sum(
                    1
                    for result in results
                    if result["path_changed"]
                )
                / len(results)
                * 100
            )

            correct_percentage = (
                correct_count
                / len(results)
                * 100
            )

            print()
            print(
                "Scenario:",
                scenario
            )

            print(
                "Update:",
                int(percentage_change * 100),
                "%"
            )

            print(
                "Experiments:",
                len(results)
            )

            print(
                "Average M-FARS score:",
                f"{average_score:.4f}"
            )

            print(
                "Path changed:",
                f"{path_change_percentage:.2f}%"
            )

            print(
                "Correctness:",
                f"{correct_percentage:.2f}%"
            )

            print(
                "Average M-FARS time:",
                f"{average_m_fars_time:.8f}",
                "seconds"
            )

            print(
                "Average full-Dijkstra verification time:",
                f"{average_true_time:.8f}",
                "seconds"
            )

            print(
                "Average M-FARS nodes processed:",
                f"{average_m_fars_nodes:.2f}"
            )

            print(
                "Average full-Dijkstra nodes processed:",
                f"{average_true_nodes:.2f}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("M-FARS V1 FINAL SUMMARY")
    print("=" * 70)

    if total_experiments > 0:

        overall_correctness = (
            total_correct
            / total_experiments
            * 100
        )

        print(
            "Total experiments:",
            total_experiments
        )

        print(
            "Overall correctness:",
            f"{overall_correctness:.2f}%"
        )

    print()
    print("Strategy counts:")

    for strategy, count in strategy_counts.items():

        print(
            " ",
            strategy,
            ":",
            count
        )

    print()
    print("IMPORTANT:")
    print(
        "The verification Dijkstra is intentionally run for every"
    )
    print(
        "experiment. Its time is NOT counted as M-FARS runtime."
    )
    print(
        "It is only the correctness oracle."
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    run_benchmark()