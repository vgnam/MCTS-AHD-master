import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        remaining_capacity = capacity - demands[i] if i != 0 else capacity
        for j in range(n):
            if i == j or j == 0:
                continue
            if demands[i] + demands[j] <= capacity:
                spatial_weight = 1.0 / (1.0 + distance_matrix[i, j])
                demand_weight = (demands[j] / remaining_capacity) if remaining_capacity > 0 else 0.0
                heuristics_matrix[i, j] = 0.6 * spatial_weight + 0.4 * demand_weight
            else:
                heuristics_matrix[i, j] = 0.0

    row_sums = heuristics_matrix.sum(axis=1, keepdims=True)
    heuristics_matrix = np.divide(heuristics_matrix, row_sums, where=row_sums!=0)

    return heuristics_matrix
