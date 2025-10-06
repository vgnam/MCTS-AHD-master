def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return destination_node

    def score(node):
        sum_distance_to_unvisited = sum(distance_matrix[node][unvisited_node] for unvisited_node in unvisited_nodes if unvisited_node != node)
        clustering_penalty = sum(1 / (1 + distance_matrix[node][other_node]) for other_node in unvisited_nodes if other_node != node)
        return (distance_matrix[current_node][node] / (1 + sum_distance_to_unvisited) / (1 + distance_matrix[node][destination_node])) * clustering_penalty

    next_node = min(unvisited_nodes, key=score)
    return next_node
