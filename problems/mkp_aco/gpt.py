import numpy as np

def heuristics_v2(prize, weight):
    remaining_capacity = np.ones(weight.shape[1])
    heuristics_matrix = np.zeros(weight.shape[0])
    base_ratio_power = 2.0
    adaptive_power = 1.2
    base_penalty_factor = 10.0

    for i in range(weight.shape[0]):
        normalized_weight = weight[i] / (remaining_capacity + 1e-10)
        weight_balance = 1 - np.std(normalized_weight) / (np.mean(normalized_weight) + 1e-10)

        if np.all(weight[i] <= remaining_capacity):
            heuristics_matrix[i] = (prize[i] ** base_ratio_power) * (1 + weight_balance) / (np.sum(normalized_weight ** 2) + 1e-10)
        else:
            excess = np.maximum(weight[i] - remaining_capacity, 0)
            penalty_factor = base_penalty_factor * (1 + np.sum(excess ** adaptive_power) / (np.sum(remaining_capacity) + 1e-10))
            infeasibility_penalty = 1 / (1 + np.exp(-penalty_factor * np.sum(excess)))
            heuristics_matrix[i] = (prize[i] ** base_ratio_power) * infeasibility_penalty * weight_balance / (np.sum(normalized_weight ** 2) + 1e-10)

    heuristics_matrix = (heuristics_matrix - np.min(heuristics_matrix)) / (np.max(heuristics_matrix) - np.min(heuristics_matrix) + 1e-10)
    return heuristics_matrix
