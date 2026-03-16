import numpy as np

def heuristics_v2(distance_matrix):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros_like(distance_matrix)
    num_iterations = 200 * n
    base_partial_solution_length = max(2, n // 2)
    base_memory_factor = 0.7
    base_exploration_noise = 0.6
    exploration_decay = 0.95
    neighborhood_size = max(2, n // 3)
    entropy_factor = 0.2

    for iteration in range(num_iterations):
        memory_factor = base_memory_factor * (1 + 0.3 * (iteration / num_iterations))
        exploration_weight = base_exploration_noise * (exploration_decay ** (iteration / n))
        start_node = np.random.randint(n)
        current_node = start_node
        visited = {current_node}
        path = [current_node]
        partial_solution_length = base_partial_solution_length + int(np.random.rand() * (n // 2))

        for _ in range(partial_solution_length - 1):
            candidates = [node for node in range(n) if node not in visited]
            if not candidates:
                break
            local_distances = distance_matrix[current_node]
            nearest_neighbors = np.argsort(local_distances)[:neighborhood_size]
            nearest_neighbors = [node for node in nearest_neighbors if node not in visited]
            if not nearest_neighbors:
                nearest_neighbors = candidates
            base_probabilities = np.array([1.0 / (local_distances[node] + 1e-10) for node in nearest_neighbors])
            memory_probabilities = heuristics_matrix[current_node, nearest_neighbors]
            probabilities = (1 - memory_factor) * base_probabilities + memory_factor * memory_probabilities
            probabilities += exploration_weight * np.random.rand(len(nearest_neighbors)) * (1 + entropy_factor * np.log(iteration + 1))
            probabilities /= probabilities.sum()
            next_node = np.random.choice(nearest_neighbors, p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node

        path_length = sum(distance_matrix[path[i], path[i+1]] for i in range(len(path)-1))
        for i in range(len(path) - 1):
            heuristics_matrix[path[i], path[i+1]] += 1.0 / (path_length * distance_matrix[path[i], path[i+1]])

    if np.max(heuristics_matrix) > 0:
        heuristics_matrix /= np.max(heuristics_matrix)

    return heuristics_matrix
