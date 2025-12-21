import numpy as np

def heuristics_v2(distance_matrix):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))
    mean_distance = np.mean(distance_matrix)

    for i in range(n):
        for j in range(n):
            if i != j:
                delta = distance_matrix[i][j] - mean_distance
                if delta < 0:
                    heuristics_matrix[i][j] = 1 - (delta / mean_distance) ** 2
                else:
                    heuristics_matrix[i][j] = (mean_distance / distance_matrix[i][j]) * np.exp(-delta / mean_distance)

    if np.sum(heuristics_matrix) > 0:
        heuristics_matrix /= np.sum(heuristics_matrix)

    return heuristics_matrix
