# ============================================================
# M-FARS PROJECT
# VERSION 1.2: CONTROLLED DYNAMIC UPDATE BENCHMARK
# ============================================================
#
# Purpose:
# Compare different types of graph updates:
#
# 1. RANDOM_SMALL
# 2. RANDOM_LARGE
# 3. SHORTEST_PATH_SMALL
# 4. SHORTEST_PATH_LARGE
#
# This is still a BASELINE experiment.
# M-FARS is NOT implemented yet.
# ============================================================

import random
import heapq
import time


# ============================================================
# 1. GENERATE A CONNECTED WEIGHTED GRAPH
# ============================================================

def generate_graph(number_of_nodes, number_of_edges):
    """
    Generate an undirected weighted graph.

    First creates a chain so that all nodes are connected.
    Then adds random edges until the requested edge count
    is reached.
    """

    # Maximum number of edges in an undirected graph
    maximum_edges = number_of_nodes * (number_of_nodes - 1) // 2

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

    # --------------------------------------------------------
    # Step 1: Create a chain.
    # This guarantees that every node is reachable.
    # --------------------------------------------------------

    for node in range(number_of_nodes - 1):

        u = node
        v = node + 1

        weight = random.randint(1, 100)

        graph[u][v] = weight
        graph[v][u] = weight

        edges.add(tuple(sorted((u, v))))

    # --------------------------------------------------------
    # Step 2: Add random edges.
    # --------------------------------------------------------

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

            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance
                previous[neighbor] = current_node

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbor)
                )

    # --------------------------------------------------------
    # Reconstruct path
    # --------------------------------------------------------

    if distances[destination] == float("inf"):

        return (
            float("inf"),
            [],
            nodes_processed,
            edge_relaxations
        )

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
# 3. GET ALL GRAPH EDGES
# ============================================================

def get_all_edges(graph):

    edges = []

    for u in graph:

        for v in graph[u]:

            # Avoid storing both (u,v) and (v,u)
            if u < v:
                edges.append((u, v))

    return edges


# ============================================================
# 4. CHANGE EDGE WEIGHT
# ============================================================

def change_edge_weight(graph, u, v, percentage_change):

    old_weight = graph[u][v]

    # At least +1 so the edge actually changes
    change = max(
        1,
        int(old_weight * percentage_change)
    )

    new_weight = old_weight + change

    graph[u][v] = new_weight
    graph[v][u] = new_weight

    return old_weight, new_weight


# ============================================================
# 5. SELECT RANDOM EDGE
# ============================================================

def choose_random_edge(graph):

    edges = get_all_edges(graph)

    if not edges:
        return None

    return random.choice(edges)


# ============================================================
# 6. SELECT AN EDGE FROM THE SHORTEST PATH
# ============================================================

def choose_shortest_path_edge(path):

    """
    Select one edge from the current shortest path.

    We avoid the first/last node and simply choose a random
    edge from the path when the path has at least 2 edges.
    """

    if len(path) < 2:
        return None

    path_edges = []

    for i in range(len(path) - 1):

        u = path[i]
        v = path[i + 1]

        path_edges.append((u, v))

    return random.choice(path_edges)


# ============================================================
# 7. CHECK WHETHER EDGE IS ON A PATH
# ============================================================

def edge_is_on_path(edge, path):

    u, v = edge

    target = tuple(sorted((u, v)))

    for i in range(len(path) - 1):

        path_edge = tuple(
            sorted((path[i], path[i + 1]))
        )

        if path_edge == target:
            return True

    return False


# ============================================================
# 8. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
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
    # Choose edge according to scenario
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
    # Was the selected edge on the shortest path?
    # --------------------------------------------------------

    was_on_shortest_path = edge_is_on_path(
        selected_edge,
        initial_path
    )

    # --------------------------------------------------------
    # Apply update
    # --------------------------------------------------------

    old_weight, new_weight = change_edge_weight(
        graph,
        u,
        v,
        percentage_change
    )

    # --------------------------------------------------------
    # Run Dijkstra AFTER update
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
    # Check whether path changed
    # --------------------------------------------------------

    path_changed = (
        initial_path != new_path
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "nodes": number_of_nodes,
        "edges": number_of_edges,
        "scenario": scenario,
        "update_percent": percentage_change * 100,
        "old_weight": old_weight,
        "new_weight": new_weight,
        "edge_on_shortest_path": was_on_shortest_path,
        "path_changed": path_changed,
        "distance": new_distance,
        "nodes_processed": new_nodes,
        "edge_relaxations": new_relaxations,
        "time": execution_time
    }


# ============================================================
# 9. PRINT RESULTS FOR ONE SCENARIO
# ============================================================

def print_scenario_result(
    scenario,
    percentage_change,
    results
):

    if not results:
        print("No successful experiments.")
        return

    total_time = sum(
        result["time"]
        for result in results
    )

    total_nodes = sum(
        result["nodes_processed"]
        for result in results
    )

    total_relaxations = sum(
        result["edge_relaxations"]
        for result in results
    )

    path_changed_count = sum(
        1
        for result in results
        if result["path_changed"]
    )

    edge_on_path_count = sum(
        1
        for result in results
        if result["edge_on_shortest_path"]
    )

    count = len(results)

    average_time = total_time / count
    average_nodes = total_nodes / count
    average_relaxations = total_relaxations / count

    path_changed_percent = (
        path_changed_count / count
    ) * 100

    edge_on_path_percent = (
        edge_on_path_count / count
    ) * 100

    print()
    print("Scenario:", scenario)
    print(
        "Update:",
        int(percentage_change * 100),
        "%"
    )
    print(
        "Experiments:",
        count
    )
    print(
        "Edge was on shortest path:",
        f"{edge_on_path_percent:.2f}%"
    )
    print(
        "Path changed:",
        f"{path_changed_percent:.2f}%"
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


# ============================================================
# 10. MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 70)
    print("M-FARS VERSION 1.2")
    print("CONTROLLED DYNAMIC UPDATE BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Graph sizes
    # --------------------------------------------------------

    graph_sizes = [
        100,
        500,
        1000,
        5000
    ]

    # --------------------------------------------------------
    # Number of experiments for each condition
    # --------------------------------------------------------

    number_of_experiments = 10

    # --------------------------------------------------------
    # Four update scenarios
    # --------------------------------------------------------

    scenarios = [
        ("RANDOM_SMALL", 0.05),
        ("RANDOM_LARGE", 0.50),
        ("SHORTEST_PATH_SMALL", 0.05),
        ("SHORTEST_PATH_LARGE", 0.50)
    ]

    print("\nStarting controlled experiments...\n")

    # ========================================================
    # GRAPH SIZE LOOP
    # ========================================================

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

        # ====================================================
        # SCENARIO LOOP
        # ====================================================

        for scenario, percentage_change in scenarios:

            results = []

            # ------------------------------------------------
            # Repeat experiment
            # ------------------------------------------------

            for experiment in range(
                number_of_experiments
            ):

                # Reproducible but different graphs
                random.seed(
                    100000
                    + number_of_nodes * 100
                    + experiment
                )

                result = run_experiment(
                    number_of_nodes,
                    number_of_edges,
                    scenario,
                    percentage_change
                )

                if result is not None:
                    results.append(result)

            # ------------------------------------------------
            # Print averages
            # ------------------------------------------------

            print_scenario_result(
                scenario,
                percentage_change,
                results
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()