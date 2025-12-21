def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if not unvisited_nodes:
        return destination_node

    total_unvisited = len(unvisited_nodes)
    remaining_nodes = total_unvisited + 1
    weight_factor = 1.0 * (total_unvisited / remaining_nodes) + 0.5

    def get_adaptive_distance(node):
        if total_unvisited == 1:
            return distance_matrix[current_node][node]
        mean = sum(distance_matrix[node][n] for n in unvisited_nodes if n != node) / (total_unvisited - 1)
        variance = sum((distance_matrix[node][n] - mean) ** 2 for n in unvisited_nodes if n != node) / (total_unvisited - 1)
        std_dev = variance ** 0.5 if variance > 0 else 1.0
        curvature_factor = 1.0 + 0.4 * (1.0 - (total_unvisited / remaining_nodes))
        return (distance_matrix[current_node][node] - mean) / (std_dev * curvature_factor)

    def get_contrastive_bonus(node):
        if len(unvisited_nodes) < 2:
            return 0
        avg_distance = sum(distance_matrix[node][n] for n in unvisited_nodes if n != node) / (total_unvisited - 1)
        momentum_factor = 1.0 - 0.3 * (1.0 - (total_unvisited / remaining_nodes))
        curvature_bonus = 0.2 * (1.0 - (total_unvisited / remaining_nodes))
        return momentum_factor * avg_distance * (1.0 + curvature_bonus)

    def get_gnn_prior(node):
        if total_unvisited == 1:
            return 1.0
        centrality = sum(distance_matrix[node][n] for n in unvisited_nodes if n != node) / (total_unvisited - 1)
        return 1.0 / (1.0 + centrality)

    def get_angular_refinement(node):
        if total_unvisited == 1:
            return 0.0
        vector_current = [distance_matrix[current_node][n] for n in unvisited_nodes if n != node]
        vector_node = [distance_matrix[node][n] for n in unvisited_nodes if n != node]
        dot_product = sum(vc * vn for vc, vn in zip(vector_current, vector_node))
        norm_current = sum(vc ** 2 for vc in vector_current) ** 0.5
        norm_node = sum(vn ** 2 for vn in vector_node) ** 0.5
        if norm_current * norm_node == 0:
            return 0.0
        angle = dot_product / (norm_current * norm_node)
        return angle

    def score(node):
        adaptive_local = get_adaptive_distance(node)
        global_score = distance_matrix[node][destination_node]
        contrastive_bonus = get_contrastive_bonus(node)
        curvature_weight = 0.4 * (1.0 - (total_unvisited / remaining_nodes))
        gnn_prior = get_gnn_prior(node)
        angular_refinement = get_angular_refinement(node)
        return (weight_factor * adaptive_local) + ((1 - weight_factor) * global_score) - (0.3 * contrastive_bonus) + (curvature_weight * (adaptive_local - global_score)) + (0.5 * gnn_prior) + (0.2 * angular_refinement)

    next_node = min(unvisited_nodes, key=score)
    return next_node
