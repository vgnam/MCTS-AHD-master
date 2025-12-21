import numpy as np

def heuristics(edge_distance, local_opt_tour, edge_n_used):
    n = len(local_opt_tour)
    updated_edge_distance = edge_distance.copy()
    current_iteration = np.max(edge_n_used) if np.max(edge_n_used) > 0 else 1
    max_iterations = 1000
    max_usage = np.max(edge_n_used) if np.max(edge_n_used) > 0 else 1

    progress = np.sum(edge_n_used > 0) / (n * n)
    decay_factor = 0.9 ** (current_iteration / max_iterations) * (1 - 0.7 * progress)

    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        edge_n_used[u, v] += 1

    diversity_bonus = np.zeros((n, n))
    for u in range(n):
        used_edges = np.sum(edge_n_used[u, :] > 0)
        exploration_bias = 1 / (1 + used_edges) if used_edges > 0 else 1
        diversity_bonus[u, :] = 1 + exploration_bias * (1 + 0.2 * np.log(1 + current_iteration))

    stagnation_threshold = 0.3 * max_iterations
    if current_iteration > stagnation_threshold:
        stagnation_penalty = 1 - 0.7 * (current_iteration - stagnation_threshold) / (max_iterations - stagnation_threshold)
    else:
        stagnation_penalty = 1.0

    for u in range(n):
        for v in range(n):
            if u != v:
                last_usage = edge_n_used[u, v]
                if last_usage > 0:
                    recency_window = min(150, current_iteration // 7)
                    recency_score = (current_iteration - last_usage) / (recency_window + 1)

                    frequency_score = np.log(1 + last_usage) ** 0.5

                    memory_decay = 0.95 ** (current_iteration / max_iterations)
                    historical_performance = (1 / (1 + last_usage)) * memory_decay

                    updated_edge_distance[u, v] = (
                        edge_distance[u, v] *
                        (1 + 0.7 * recency_score) *
                        (1 + 1.2 * frequency_score) *
                        decay_factor *
                        (1 + 0.5 * historical_performance) *
                        diversity_bonus[u, v] *
                        stagnation_penalty
                    )
                else:
                    updated_edge_distance[u, v] = edge_distance[u, v] * diversity_bonus[u, v] * 1.5

    reweighting_period = max(3, 10 - current_iteration // 150)
    if current_iteration % reweighting_period == 0:
        for u in range(n):
            for v in range(n):
                if u != v:
                    usage_ratio = edge_n_used[u, v] / (max_usage + 1)
                    updated_edge_distance[u, v] = (
                        edge_distance[u, v] *
                        (1 + 1.0 * np.log(1 + edge_n_used[u, v])) *
                        (1 + 0.3 * (1 - usage_ratio)) *
                        diversity_bonus[u, v]
                    )

    return updated_edge_distance