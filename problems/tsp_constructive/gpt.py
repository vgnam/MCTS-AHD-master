def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return destination_node

    min_distance = float('inf')
    next_node = None
    lookahead_factor = 0.3  # Weight for future steps

    for node in unvisited_nodes:
        direct_distance = distance_matrix[current_node][node]
        if destination_node in unvisited_nodes:
            # Estimate future path cost: direct + return to destination
            future_cost = direct_distance + distance_matrix[node][destination_node]
        else:
            future_cost = direct_distance

        # Combine immediate and future cost
        total_cost = direct_distance + lookahead_factor * future_cost

        if total_cost < min_distance:
            min_distance = total_cost
            next_node = node

    return next_node
