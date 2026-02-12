import numpy as np

def select_next_item_v2(remaining_capacity, weights, values):
    log_value_to_weight = np.log(values / weights + 1e-10)
    capacity_utilization = weights / remaining_capacity
    quadratic_penalty = (capacity_utilization - 0.5)**2
    cubic_reward = (capacity_utilization - 0.5)**3
    adjusted_ratio = log_value_to_weight * (1 - quadratic_penalty) + cubic_reward
    feasible_indices = np.where(weights <= remaining_capacity)[0]
    if len(feasible_indices) == 0:
        return -1
    best_indices = feasible_indices[np.where(adjusted_ratio[feasible_indices] == np.max(adjusted_ratio[feasible_indices]))]
    if len(best_indices) == 1:
        return best_indices[0]
    else:
        tie_breaker = weights[best_indices] / values[best_indices]
        return updated_edge_distance