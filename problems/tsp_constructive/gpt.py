def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if destination_node in unvisited_nodes:
        return destination_node

    max_score = -float('inf')
    next_node = None
    total_unvisited = len(unvisited_nodes)
    centrality = {node: sum(distance_matrix[node]) for node in unvisited_nodes}
    max_centrality = max(centrality.values()) if centrality else 1
    visit_count = {node: 0 for node in unvisited_nodes}
    max_visit = max(visit_count.values()) if visit_count else 1

    for node in unvisited_nodes:
        current_to_node = distance_matrix[current_node][node]
        node_to_destination = distance_matrix[node][destination_node]

        if current_to_node == 0:
            continue

        ratio = node_to_destination / current_to_node
        adaptive_weight = (1 - (total_unvisited / (total_unvisited + 5))) * ratio * (1 + (centrality[node] / max_centrality))
        exploration_factor = 0.8 * (total_unvisited / (total_unvisited + 3)) * (1 / (1 + current_to_node)) * (1 - (visit_count[node] / (max_visit + 1)))
        connectivity_factor = 0.7 * (centrality[node] / max_centrality) * (1 + (visit_count[node] / (max_visit + 1))) * (1 - (current_to_node / (sum(distance_matrix[current_node]) + 1)))
        diversity_factor = 0.4 * (1 - (sum(distance_matrix[node]) / sum(sum(distance_matrix)))) * (1 + (visit_count[node] / (max_visit + 1))) * (1 - (node_to_destination / (sum(distance_matrix[node]) + 1)))

        score = adaptive_weight + exploration_factor + connectivity_factor + diversity_factor

        if score > max_score:
            max_score = score
            next_node = node

        visit_count[node] += 1

    return next_node
