importance, and a probabilistic sampling approach to avoid premature convergence.}

import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics_matrix[i, j] = 0.0
            else:
                # Distance-based heuristic (inverse distance)
                distance_heuristic = 1.0 / (distance_matrix[i, j] + 1e-10)

                # Demand-based heuristic (capacity-aware soft constraint)
                remaining_capacity = capacity - demands[i] if i != 0 else capacity
                demand_heuristic = np.exp(-(demands[j] / remaining_capacity) ** 2)

                # Combine heuristics
                heuristics_matrix[i, j] = distance_heuristic * demand_heuristic

    # Dynamic normalization based on node degrees
    row_sums = np.sum(heuristics_matrix, axis=1, keepdims=True)
    heuristics_matrix = heuristics_matrix / (row_sums + 1e-10)

    return heuristics_matrix
