import numpy as np

def heuristics(distance_matrix):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros_like(distance_matrix)

    # Compute inverse distance
    inv_distance = 1.0 / (distance_matrix + np.eye(n) * np.max(distance_matrix))

    # Compute local density (average distance to neighbors)
    local_density = np.mean(inv_distance, axis=1)

    # Combine inverse distance and local density
    for i in range(n):
        for j in range(i + 1, n):
            heuristics_matrix[i, j] = inv_distance[i, j] * local_density[i] * local_density[j]
            heuristics_matrix[j, i] = heuristics_matrix[i, j]

    # Normalize to [0, 1] range
    max_heuristic = np.max(heuristics_matrix)
    if max_heuristic > 0:
        heuristics_matrix = heuristics_matrix / max_heuristic

    return heuristics_matrix
