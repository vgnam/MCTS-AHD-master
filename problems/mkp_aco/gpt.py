import numpy as np

def heuristics_v2(prize, weight):
    n, m = weight.shape
    prize_normalized = prize / np.max(prize)
    weight_sum = np.sum(weight, axis=1)
    constraint_tightness = np.mean(weight, axis=0)
    global_tightness = np.sum(constraint_tightness)
    weight_balance = np.std(weight, axis=1) / (np.mean(weight, axis=1) + 1e-6)
    prize_to_weight = prize_normalized / (weight_sum + 1e-10)
    dim_penalty = np.exp(np.sum(weight * constraint_tightness, axis=1)) ** (1 + 2 * global_tightness)
    heuristics_matrix = 0.6 * prize_to_weight * (1 / (1 + weight_balance)) - 0.4 * dim_penalty
    return heuristics_matrix
