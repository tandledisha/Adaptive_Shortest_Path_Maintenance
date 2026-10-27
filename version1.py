# ============================================================
# M-FARS PROJECT
# VERSION 1: Dynamic Graph + Dijkstra Baseline
# ============================================================

import random
import heapq
import time


# ------------------------------------------------------------
# 1. GENERATE A RANDOM WEIGHTED GRAPH
# ------------------------------------------------------------

def generate_graph(number_of_nodes, number_of_edges):
    """
    Creates an undirected weighted graph.

    Example:
        0 -- 5 -- 1
        |         |
        3         7
        |         |
        2 -- 4 -- 3
    """

    graph = {node: {} for node in range(number_of_nodes)}

    edges = set()

    while len(edges) < number_of_edges:

        u = random.randint(0, number_of_nodes - 1)
        v = random.randint(0, number_of_nodes - 1)

        # Don't connect a node to itself
        if u == v:
            continue

        # Store edge in sorted form to avoid duplicates
        edge = tuple(sorted((u, v)))

        if edge in edges:
            continue

        edges.add(edge)

        # Random positive weight
        weight = random.randint(1, 100)

        # Undirected graph
        graph[u][v] = weight
        graph[v][u] = weight

    return graph


# ------------------------------------------------------------
# 2. DIJKSTRA'S SHORTEST PATH ALGORITHM
# ------------------------------------------------------------

def dijkstra(graph, source, destination):
    """
    Finds the shortest path from source to destination.

    Returns:
        shortest_distance
        shortest_path
        number_of_nodes_processed
        number_of_edge_relaxations
    """

    # Initially, distance to every node is infinity
    distances = {
        node: float("inf")
        for node in graph
    }

    # Used to reconstruct the shortest path
    previous = {
        node: None
        for node in graph
    }

    # Distance from source to itself is 0
    distances[source] = 0

    # Priority queue
    priority_queue = [(0, source)]

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

        # We reached destination
        if current_node == destination:
            break

        # Check all neighboring nodes
        for neighbor, weight in graph[current_node].items():

            edge_relaxations += 1

            new_distance = current_distance + weight

            # Found a shorter route
            if new_distance < distances[neighbor]:

                distances[neighbor] = new_distance

                previous[neighbor] = current_node

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbor)
                )

    # --------------------------------------------------------
    # Reconstruct shortest path
    # --------------------------------------------------------

    path = []

    current = destination

    if distances[destination] == float("inf"):
        return None, [], nodes_processed, edge_relaxations

    while current is not None:

        path.append(current)

        current = previous[current]

    # Path was created backwards
    path.reverse()

    return (
        distances[destination],
        path,
        nodes_processed,
        edge_relaxations
    )


# ------------------------------------------------------------
# 3. CALCULATE THE COST OF A PATH
# ------------------------------------------------------------

def calculate_path_cost(graph, path):

    if not path:
        return float("inf")

    cost = 0

    for i in range(len(path) - 1):

        current_node = path[i]
        next_node = path[i + 1]

        # Check whether edge exists
        if next_node not in graph[current_node]:
            return float("inf")

        cost += graph[current_node][next_node]

    return cost


# ------------------------------------------------------------
# 4. FIND A RANDOM EXISTING EDGE
# ------------------------------------------------------------

def get_random_edge(graph):

    all_edges = []

    for u in graph:

        for v in graph[u]:

            # Add only one direction
            if u < v:
                all_edges.append((u, v))

    if not all_edges:
        return None

    return random.choice(all_edges)


# ------------------------------------------------------------
# 5. CHANGE EDGE WEIGHT
# ------------------------------------------------------------

def change_edge_weight(graph, u, v, new_weight):

    if v not in graph[u]:
        return False

    old_weight = graph[u][v]

    graph[u][v] = new_weight
    graph[v][u] = new_weight

    print("\nGraph Update")
    print("---------------------------")
    print("Changed edge :", (u, v))
    print("Old weight   :", old_weight)
    print("New weight   :", new_weight)

    return True


# ------------------------------------------------------------
# 6. MAIN EXPERIMENT
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("M-FARS VERSION 1")
    print("Dynamic Graph + Dijkstra Baseline")
    print("=" * 60)

    # --------------------------------------------------------
    # EXPERIMENT SETTINGS
    # --------------------------------------------------------

    NUMBER_OF_NODES = 100
    NUMBER_OF_EDGES = 300

    SOURCE = 0
    DESTINATION = NUMBER_OF_NODES - 1

    # Same random graph every time
    # This makes our experiment reproducible
    random.seed(42)

    # --------------------------------------------------------
    # STEP 1: GENERATE GRAPH
    # --------------------------------------------------------

    print("\n[1] Generating graph...")

    graph = generate_graph(
        NUMBER_OF_NODES,
        NUMBER_OF_EDGES
    )

    print("Number of nodes :", NUMBER_OF_NODES)
    print("Number of edges :", NUMBER_OF_EDGES)

    # --------------------------------------------------------
    # STEP 2: INITIAL DIJKSTRA
    # --------------------------------------------------------

    print("\n[2] Running initial Dijkstra...")

    start_time = time.perf_counter()

    initial_distance, initial_path, initial_nodes, initial_relaxations = dijkstra(
        graph,
        SOURCE,
        DESTINATION
    )

    initial_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # CHECK WHETHER PATH EXISTS
    # --------------------------------------------------------

    if not initial_path:

        print("\nNo path exists between source and destination.")

        return

    print("\nInitial shortest path:")
    print(initial_path)

    print("Initial distance       :", initial_distance)
    print("Nodes processed       :", initial_nodes)
    print("Edge relaxations      :", initial_relaxations)
    print("Execution time        :", initial_time, "seconds")

    # Verify path cost
    calculated_cost = calculate_path_cost(
        graph,
        initial_path
    )

    print("Verified path cost    :", calculated_cost)

    # --------------------------------------------------------
    # STEP 3: SELECT RANDOM EDGE
    # --------------------------------------------------------

    selected_edge = get_random_edge(graph)

    if selected_edge is None:

        print("No edge available for update.")

        return

    u, v = selected_edge

    old_weight = graph[u][v]

    # Increase edge weight by 10
    new_weight = old_weight + 10

    # --------------------------------------------------------
    # STEP 4: MODIFY GRAPH
    # --------------------------------------------------------

    change_edge_weight(
        graph,
        u,
        v,
        new_weight
    )

    # --------------------------------------------------------
    # STEP 5: RUN DIJKSTRA AGAIN
    # --------------------------------------------------------

    print("\n[3] Running Dijkstra after graph update...")

    start_time = time.perf_counter()

    new_distance, new_path, new_nodes, new_relaxations = dijkstra(
        graph,
        SOURCE,
        DESTINATION
    )

    new_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # STEP 6: DISPLAY NEW RESULT
    # --------------------------------------------------------

    if not new_path:

        print("\nNo path exists after update.")

        return

    print("\nNew shortest path:")
    print(new_path)

    print("New distance          :", new_distance)
    print("Nodes processed       :", new_nodes)
    print("Edge relaxations      :", new_relaxations)
    print("Execution time        :", new_time, "seconds")

    # --------------------------------------------------------
    # STEP 7: VALIDATE RESULT
    # --------------------------------------------------------

    verified_new_cost = calculate_path_cost(
        graph,
        new_path
    )

    print("Verified path cost    :", verified_new_cost)

    if verified_new_cost == new_distance:

        print("\nRESULT: Correct shortest path")

    else:

        print("\nRESULT: ERROR in shortest path calculation")


# ------------------------------------------------------------
# PROGRAM START
# ------------------------------------------------------------

if __name__ == "__main__":
    main()