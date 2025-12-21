import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))
    avg_demand = np.mean(demand)
    std_demand = np.std(demand)
    dynamic_threshold = max(0.65 * capacity, 0.8 * capacity - 0.4 * avg_demand + 0.15 * std_demand)
    optimal_sum = capacity
    size_weights = demand / np.max(demand)

    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics_matrix[i][j] = 0.0
            else:
                pair_sum = demand[i] + demand[j]
                if pair_sum < dynamic_threshold or pair_sum > capacity:
                    heuristics_matrix[i][j] = 0.0
                else:
                    diff = abs(pair_sum - optimal_sum)
                    imbalance_penalty = (abs(demand[i] - demand[j]) / avg_demand) ** 1.5
                    if diff == 0:
                        heuristics_matrix[i][j] = 1.0
                    else:
                        size_ratio = min(demand[i], demand[j]) / max(demand[i], demand[j])
                        scaling_factor = np.log(1 + size_ratio) ** 2
                        heuristics_matrix[i][j] = (1.0 / (diff ** 1.5)) * scaling_factor * (1.0 - imbalance_penalty)

    return heuristics_matrix
