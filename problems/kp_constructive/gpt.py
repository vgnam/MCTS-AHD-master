import numpy as np
def select_next_item(remaining_capacity, weights, values):
    ratios = values / weights
    capacity_factor = np.log(1 + (weights / remaining_capacity))
    hybrid_score = ratios * (1 + 0.3 * capacity_factor)
    sorted_indices = np.argsort(-hybrid_score)
    for idx in sorted_indices:
        if weights[idx] <= remaining_capacity:
            return idx
    return result