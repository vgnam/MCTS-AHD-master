def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return destination_node

    candidates = unvisited_nodes.copy()
    if destination_node in candidates:
        candidates.remove(destination_node)

    if not candidates:
        return destination_node

    distances = [distance_matrix[current_node][node] for node in candidates]
    min_dist = min(distances)
    max_dist = max(distances)

    if min_dist == max_dist:
        weights = [1.0 / len(candidates)] * len(candidates)
    else:
        normalized = [(max_dist - d) / (max_dist - min_dist) for d in distances]
        weights = [w / sum(normalized) for w in normalized]

    if destination_node in unvisited_nodes:
        dest_dist = distance_matrix[current_node][destination_node]
        if min_dist == max_dist:
            dest_weight = 1.0 / (len(candidates) + 1)
        else:
            dest_weight = (max_dist - dest_dist) / (max_dist - min_dist) / sum(normalized)
        candidates.append(destination_node)
        weights.append(dest_weight)
        weights = [w / sum(weights) for w in weights]

    next_node = random.choices(candidates, weights=weights, k=1)[0]

    return next_node
