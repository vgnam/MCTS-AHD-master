import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))

    # Inverse squared distance heuristic
    inv_squared_distance = 1 / (distance_matrix ** 2 + 1e-6)

    # Angular alignment with depot (node 0)
    depot_coord = coordinates[0]
    vectors = coordinates - depot_coord
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diff = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # Wrap around 2pi
    normalized_angle_diff = angle_diff / np.pi  # Normalize to [0, 1]

    # Dynamic demand compatibility heuristic (adaptive to remaining capacity)
    demand_compatibility = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and i != 0 and j != 0:  # Skip depot and self-loops
                total_demand = demands[i] + demands[j]
                if total_demand <= capacity:
                    demand_compatibility[i, j] = 1 - (total_demand / capacity) ** 2  # Stronger penalty for near-capacity edges
                else:
                    demand_compatibility[i, j] = 0  # Hard constraint for overloading

    # Normalize distance to [0, 1]
    max_distance = np.max(distance_matrix)
    normalized_distance = distance_matrix / max_distance if max_distance > 0 else distance_matrix

    # Multi-objective optimization: prioritize edges with high distance score, angular alignment, and demand compatibility
    heuristics_matrix = inv_squared_distance * (1 - normalized_angle_diff) * (demand_compatibility + 1e-6) * (1 - normalized_distance)

    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(heuristics_matrix, 0)

    return heuristics_matrix
