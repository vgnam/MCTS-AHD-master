import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))

    depot_x, depot_y = coordinates[0]
    distances_from_depot = np.linalg.norm(coordinates - [depot_x, depot_y], axis=1)

    demand_balance = (demands[None, :] + demands[:, None]) / capacity
    balance_penalty = 1.0 / (1.0 + np.abs(demand_balance - 0.7) * 1.5)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            distance_factor = 0.5 / (1.0 + distance_matrix[i, j] / np.median(distance_matrix[i, :]))

            spatial_factor = 0.3 * (1.0 - (distances_from_depot[i] + distances_from_depot[j]) / (2 * np.median(distances_from_depot)))

            demand_factor = 0.8 * balance_penalty[i, j]

            heuristics_matrix[i, j] = distance_factor * spatial_factor * demand_factor

    heuristics_matrix = (heuristics_matrix - np.min(heuristics_matrix)) / (np.max(heuristics_matrix) - np.min(heuristics_matrix) + 1e-10)
    heuristics_matrix = np.power(heuristics_matrix, 1.5)

    return heuristics_matrix
