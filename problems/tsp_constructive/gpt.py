def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return destination_node

    next_node = None
    best_score = float('inf')

    for node in unvisited_nodes:
        immediate_distance = distance_matrix[current_node][node]

        if len(unvisited_nodes) > 1:
            remaining_nodes = unvisited_nodes - {node}
            avg_future_distance = sum(distance_matrix[node][n] for n in remaining_nodes) / len(remaining_nodes)
        else:
            avg_future_distance = 0

        total_score = immediate_distance + 0.5 * avg_future_distance

        if total_score < best_score:
            best_score = total_score
            next_node = node
        elif total_score == best_score and node == destination_node:
            next_node = node

    return next_node
