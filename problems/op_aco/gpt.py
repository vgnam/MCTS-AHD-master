import numpy as np

def heuristics_v2(prize, distance, maxlen):
    n = len(prize)
    heuristics_matrix = np.zeros((n, n))
    max_prize = np.max(prize)
    mean_prize = np.mean(prize)

    # Compute local prize density and global prize distribution
    prize_density = np.zeros(n)
    global_prize_dist = np.zeros(n)
    for i in range(n):
        neighbors = np.where(distance[i] <= maxlen)[0]
        if len(neighbors) > 1:
            prize_density[i] = np.mean(prize[neighbors])
            global_prize_dist[i] = np.sum(prize[neighbors] * np.exp(-distance[i][neighbors] / maxlen))

    # Compute node popularity scores with adaptive weights
    temp_heuristics = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and distance[i][j] <= maxlen:
                temp_heuristics[i][j] = prize[j] / (distance[i][j] * (1 + 0.1 * (distance[i][j] / maxlen)))

    node_popularity = np.sum(temp_heuristics, axis=0) + np.sum(temp_heuristics, axis=1)
    node_popularity = node_popularity / (np.sum(node_popularity) + 1e-10)

    # Calculate median and quantile-based thresholds for dynamic scaling
    valid_ratios = []
    for i in range(n):
        for j in range(n):
            if i != j and distance[i][j] <= maxlen:
                valid_ratios.append(prize[j] / (distance[i][j] * (1 + 0.1 * (distance[i][j] / maxlen))))

    median_ratio = np.median(valid_ratios) if valid_ratios else 1.0
    upper_quantile = np.percentile(valid_ratios, 75) if valid_ratios else 1.0

    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics_matrix[i][j] = 0
            else:
                if distance[i][j] > maxlen:
                    heuristics_matrix[i][j] = 0
                else:
                    ratio = prize[j] / (distance[i][j] * (1 + 0.1 * (distance[i][j] / maxlen)))
                    prize_weight = prize[j] * (1 + 0.5 * (prize_density[j] / mean_prize) + 0.3 * (global_prize_dist[j] / (mean_prize * n)))
                    popularity_factor = node_popularity[j] * (1 + 0.2 * (prize_density[j] / mean_prize))

                    if ratio > median_ratio:
                        distance_penalty = (distance[i][j] / maxlen) + 0.3 * (distance[i][j] / maxlen)**2 + 0.1 * (distance[i][j] / maxlen)**3
                        heuristic_val = (prize_weight / distance[i][j]) * (ratio / median_ratio) ** (1.5 if ratio > upper_quantile else 1.0) * np.exp(-distance_penalty) * popularity_factor
                        heuristics_matrix[i][j] = heuristic_val
                    else:
                        heuristics_matrix[i][j] = 0

    # Normalization with adaptive scaling
    max_heuristic = np.max(heuristics_matrix)
    if max_heuristic > 0:
        heuristics_matrix /= max_heuristic
        heuristics_matrix *= (max_prize / mean_prize) * (1 + 0.3 * (max_prize / mean_prize))

    return heuristics_matrix
