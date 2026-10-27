# ============================================================
# M-FARS PROJECT
# VERSION 1.1: AUTOMATED BASELINE BENCHMARK
# ============================================================

import random
import heapq
import time


# ============================================================
# 1. GENERATE GRAPH
# ============================================================

def generate_graph(number_of_nodes, number_of_edges):

    graph = {
        node: {}
        for node in range(number_of_nodes)
    }

    edges = set()

    while len(edges) < number_of_edges:

        u = random.randint(0, number_of_nodes - 1)
        v = random.randint(0, number_of_nodes - 1)

        # Don't connect node to itself
        if u == v:
            continue

        # Store edge only once
        edge = tuple(sorted((u, v)))

        if edge in edges:
            continue

        edges.add(edge)

        # Random positive weight
        weight = random.randint(1, 100)

        graph[u][v] = weight
        graph[v][u] = weight

    return graph


# ============================================================
# 2. DIJKSTRA
# ============================================================

def dijkstra(graph, source, destination):

    distances = {
        node: float("inf")
        for node in graph
    }

    previous = {
        node: None
        for node in graph
    }

    distances[source] = 0

    priority_queue = [
        (0, source)
    ]

    visited = set()

    nodes_processed = 0
    edge_relaxations = 0

    while priority_queue:

        current_distance, current_node = heapq.heappop(
            priority_queue
        )

        # Ignore already processed nodes
        if current_node in visited:
            continue

        visited.add(current_node)

        nodes_processed += 1

        # Destination reached
        if current_node == destination:
            break

        # Check neighbors
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

    # ========================================================
    # RECONSTRUCT PATH
    # ========================================================

    path = []

    if distances[destination] == float("inf"):

        return (
            float("inf"),
            [],
            nodes_processed,
            edge_relaxations
        )

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
# 3. GET RANDOM EXISTING EDGE
# ============================================================

def get_random_edge(graph):

    edges = []

    for u in graph:

        for v in graph[u]:

            # Only store one direction
            if u < v:
                edges.append((u, v))

    if len(edges) == 0:
        return None

    return random.choice(edges)


# ============================================================
# 4. CHANGE EDGE WEIGHT
# ============================================================

def change_edge_weight(
    graph,
    u,
    v,
    percentage_change
):

    old_weight = graph[u][v]

    # Calculate new weight
    change = max(
        1,
        int(old_weight * percentage_change)
    )

    new_weight = old_weight + change

    graph[u][v] = new_weight
    graph[v][u] = new_weight

    return old_weight, new_weight


# ============================================================
# 5. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    number_of_nodes,
    number_of_edges,
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
    # Initial Dijkstra
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

    # If no path exists, skip experiment
    if not initial_path:

        return None

    # --------------------------------------------------------
    # Select random edge
    # --------------------------------------------------------

    selected_edge = get_random_edge(graph)

    if selected_edge is None:

        return None

    u, v = selected_edge

    # --------------------------------------------------------
    # Update graph
    # --------------------------------------------------------

    old_weight, new_weight = change_edge_weight(
        graph,
        u,
        v,
        percentage_change
    )

    # --------------------------------------------------------
    # Dijkstra AFTER update
    # --------------------------------------------------------

    start_time = time.perf_counter()

    (
        new_distance,
        new_path,
        new_nodes,
        new_relaxations
    ) = dijkstra(
        graph,
        source,
        destination
    )

    execution_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Did shortest path change?
    # --------------------------------------------------------

    path_changed = (
        initial_path != new_path
    )

    # --------------------------------------------------------
    # Return experiment results
    # --------------------------------------------------------

    return {
        "nodes": number_of_nodes,
        "edges": number_of_edges,

        "old_weight": old_weight,
        "new_weight": new_weight,

        "distance": new_distance,

        "path_changed": path_changed,

        "nodes_processed": new_nodes,

        "edge_relaxations": new_relaxations,

        "time": execution_time
    }


# ============================================================
# 6. MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 70)

    print(
        "M-FARS VERSION 1.1"
    )

    print(
        "AUTOMATED DIJKSTRA BASELINE BENCHMARK"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # GRAPH SIZES
    # --------------------------------------------------------

    graph_sizes = [
        100,
        500,
        1000,
        5000
    ]

    # --------------------------------------------------------
    # Number of experiments per graph size
    # --------------------------------------------------------

    number_of_experiments = 10

    # --------------------------------------------------------
    # Percentage of weight increase
    # --------------------------------------------------------

    update_percentages = [
        0.05,
        0.10,
        0.25,
        0.50
    ]

    print("\nStarting experiments...\n")

    # ========================================================
    # LOOP THROUGH GRAPH SIZES
    # ========================================================

    for number_of_nodes in graph_sizes:

        # ----------------------------------------------------
        # Choose number of edges
        # ----------------------------------------------------

        # Approximately 3 edges per node
        number_of_edges = number_of_nodes * 3

        print(
            "\n"
            + "-" * 70
        )

        print(
            "Graph size:",
            number_of_nodes,
            "nodes"
        )

        print(
            "Edges:",
            number_of_edges
        )

        print(
            "-" * 70
        )

        # ====================================================
        # LOOP THROUGH UPDATE PERCENTAGES
        # ====================================================

        for percentage_change in update_percentages:

            total_time = 0

            total_nodes = 0

            total_relaxations = 0

            successful_experiments = 0

            path_changed_count = 0

            # ------------------------------------------------
            # Run multiple experiments
            # ------------------------------------------------

            for experiment in range(
                number_of_experiments
            ):

                # Different random graph
                random.seed(
                    1000
                    + number_of_nodes
                    + experiment
                )

                result = run_experiment(
                    number_of_nodes,
                    number_of_edges,
                    percentage_change
                )

                # Experiment failed
                if result is None:
                    continue

                successful_experiments += 1

                total_time += result["time"]

                total_nodes += result[
                    "nodes_processed"
                ]

                total_relaxations += result[
                    "edge_relaxations"
                ]

                if result["path_changed"]:
                    path_changed_count += 1

            # ------------------------------------------------
            # Calculate averages
            # ------------------------------------------------

            if successful_experiments == 0:

                print(
                    "No successful experiments."
                )

                continue

            average_time = (
                total_time
                / successful_experiments
            )

            average_nodes = (
                total_nodes
                / successful_experiments
            )

            average_relaxations = (
                total_relaxations
                / successful_experiments
            )

            path_change_percentage = (
                path_changed_count
                / successful_experiments
                * 100
            )

            # ------------------------------------------------
            # Display results
            # ------------------------------------------------

            print()

            print(
                "Update:",
                int(percentage_change * 100),
                "%"
            )

            print(
                "Experiments:",
                successful_experiments
            )

            print(
                "Average time:",
                f"{average_time:.8f}",
                "seconds"
            )

            print(
                "Average nodes processed:",
                f"{average_nodes:.2f}"
            )

            print(
                "Average edge relaxations:",
                f"{average_relaxations:.2f}"
            )

            print(
                "Path changed:",
                f"{path_change_percentage:.2f}%"
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()