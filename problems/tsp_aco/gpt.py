import numpy as np

def heuristics_v2(distance_matrix):
    n = len(distance_matrix)
    heuristics_matrix = np.zeros_like(distance_matrix)
    num_samples = 500
    edge_weights = np.ones((n, n))
    temperature = 1.0
    decay_rate = 0.95
    exploration_factor = 0.1

    for _ in range(num_samples):
        path = [0]
        remaining = set(range(1, n))
        while remaining:
            last = path[-1]
            candidates = list(remaining)
            probs = np.exp((edge_weights[last, candidates] - np.max(edge_weights[last, candidates])) / temperature)
            probs = (1 - exploration_factor) * probs + exploration_factor * (1 / len(candidates))
            probs /= np.sum(probs)
            next_node = np.random.choice(candidates, p=probs)
            path.append(next_node)
            remaining.remove(next_node)
        path.append(0)

        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 2, min(i + 10, n)):
                    a, b, c, d = path[i], path[i+1], path[j], path[j+1]
                    current_dist = distance_matrix[a, b] + distance_matrix[c, d]
                    new_dist = distance_matrix[a, c] + distance_matrix[b, d]
                    if new_dist < current_dist:
                        path[i+1:j+1] = path[j:i:-1]
                        improved = True

        total_dist = sum(distance_matrix[path[i], path[i+1]] for i in range(n))
        for i in range(n):
            u, v = path[i], path[i+1]
            heuristics_matrix[u, v] += 1 / (total_dist * (1 + np.exp(-(edge_weights[u, v] - 0.5))))

        for i in range(n):
            u, v = path[i], path[i+1]
            edge_weights[u, v] *= 0.8
            edge_weights[u, v] += 1 / (total_dist * (1 + np.exp(-(edge_weights[u, v] - 0.5))))

        temperature *= decay_rate
        exploration_factor = max(0.05, exploration_factor * 0.99)

    exp_heuristics = np.exp(heuristics_matrix - np.max(heuristics_matrix))
    heuristics_matrix = exp_heuristics / np.sum(exp_heuristics)
    return heuristics_matrix
