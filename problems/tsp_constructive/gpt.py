import random
import math

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return destination_node

    candidates = list(unvisited_nodes)
    n = len(candidates)

    # Calculate nearest neighbor distance for each candidate
    nn_distances = [distance_matrix[current_node][node] for node in candidates]

    # Calculate diversity factor: sum of distances to top 3 nearest neighbors (or all if n < 3)
    diversity_factors = []
    for i, node in enumerate(candidates):
        # Find top 3 nearest neighbors (excluding self)
        neighbors = sorted([distance_matrix[node][other] for other in candidates if other != node])
        diversity = sum(neighbors[:min(3, len(neighbors))])
        diversity_factors.append(diversity)

    # Adaptive weighting based on problem characteristics
    avg_nn = sum(nn_distances) / n
    avg_diversity = sum(diversity_factors) / n
    if avg_diversity == 0:
        proximity_weight = 1.0
    else:
        proximity_weight = min(0.9, max(0.1, 0.7 * (avg_nn / (avg_nn + avg_diversity))))

    # Combine scores
    scores = []
    for i in range(n):
        score = (1 - proximity_weight) * nn_distances[i] + proximity_weight * diversity_factors[i]
        scores.append(score)

    # Determine if we should use deterministic nearest-neighbor (10% chance)
    if random.random() < 0.1 or n <= 3:
        next_node = candidates[scores.index(min(scores))]
    else:
        # Normalize scores to use as weights
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            weights = [1.0 for _ in scores]
        else:
            weights = [(max_score - score + min_score) / (max_score - min_score) for score in scores]
        next_node = random.choices(candidates, weights=weights, k=1)[0]

    return next_node
