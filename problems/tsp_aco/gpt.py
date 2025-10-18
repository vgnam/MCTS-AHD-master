import numpy as np

def heuristics_v2(distance_matrix):
    epsilon = 1e-10
    inv_dist = np.exp(-distance_matrix) / (distance_matrix + epsilon)
    k = min(3, distance_matrix.shape[0] - 1)
    sorted_dist = np.sort(distance_matrix, axis=1)
    local_density = 1.0 / (np.mean(1.0 / (sorted_dist[:, 1:k+1] + epsilon), axis=1) + epsilon)
    heuristics_matrix = inv_dist * np.outer(local_density, local_density)
    heuristics_matrix = (heuristics_matrix - np.min(heuristics_matrix)) / (np.max(heuristics_matrix) - np.min(heuristics_matrix))
    return heuristics_matrix
