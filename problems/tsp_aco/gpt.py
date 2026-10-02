import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    eps = 1e-12

    # ----- MST baseline (Prim) -----
    in_mst = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = -np.ones(n, dtype=int)
    key[0] = 0.0

    for _ in range(n):
        u = np.argmin(np.where(in_mst, np.inf, key))
        in_mst[u] = True
        row = distance_matrix[u]
        mask = ~in_mst
        better = row < key
        update = mask & better
        key[update] = row[update]
        parent[update] = u

    mst_matrix = np.zeros((n, n), dtype=float)
    for v in range(1, n):
        u = parent[v]
        if u >= 0:
            mst_matrix[v, u] = 1.0
            mst_matrix[u, v] = 1.0

    # ----- ACO parameters -----
    num_iters = 80
    ants_per_iter = 15
    rho = 0.3
    Q = 2.0
    alpha = 1.5
    beta = 2.0
    start_temp = 8.0
    end_temp = 0.5

    pheromone = np.full((n, n), 0.1, dtype=float)
    heuristic = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0.0)

    rng = np.random.default_rng()
    best_len = np.inf
    best_tour = None

    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    a, b = tour[i - 1], tour[i]
                    c, d = tour[j], tour[j + 1]
                    if distance_matrix[a, b] + distance_matrix[c, d] > distance_matrix[a, c] + distance_matrix[b, d] + eps:
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        improved = True
        return tour

    for it in range(num_iters):
        temp = start_temp - (start_temp - end_temp) * (it / (num_iters - 1))
        deposit = np.zeros((n, n), dtype=float)

        for _ in range(ants_per_iter):
            start = rng.integers(n)
            visited = np.zeros(n, dtype=bool)
            visited[start] = True
            tour = [start]
            cur = start

            while len(tour) < n:
                cand = np.where(~visited)[0]
                tau = pheromone[cur, cand] ** alpha
                eta = heuristic[cur, cand] ** beta
                weight = np.exp(-distance_matrix[cur, cand] / (temp + eps))
                prob = tau * eta * weight
                s = prob.sum()
                if s == 0:
                    prob = np.full_like(prob, 1.0 / len(cand))
                else:
                    prob /= s
                nxt = rng.choice(cand, p=prob)
                visited[nxt] = True
                tour.append(nxt)
                cur = nxt

            tour.append(start)
            tour = two_opt(tour)

            length = 0.0
            for i in range(len(tour) - 1):
                length += distance_matrix[tour[i], tour[i + 1]]

            if length < best_len:
                best_len = length
                best_tour = list(tour)

            inc = Q / (length + eps)
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                deposit[a, b] += inc
                deposit[b, a] += inc

        pheromone *= (1.0 - rho)
        pheromone += deposit

        if best_tour is not None:
            elite_inc = Q / (best_len + eps)
            for i in range(len(best_tour) - 1):
                a, b = best_tour[i], best_tour[i + 1]
                pheromone[a, b] += elite_inc
                pheromone[b, a] += elite_inc

    mst_norm = mst_matrix / (mst_matrix.sum() + eps)
    pher_norm = pheromone / (pheromone.sum() + eps)
    combined = 0.35 * mst_norm + 0.65 * pher_norm

    np.fill_diagonal(combined, 0.0)
    total = combined.sum()
    if total > 0:
        combined /= total

    return combined