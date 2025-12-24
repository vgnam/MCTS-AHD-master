import numpy as np
from collections import defaultdict

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    num_samples = 100
    edge_counts = defaultdict(int)
    total_distance = defaultdict(float)

    for _ in range(num_samples):
        # Initialize a random permutation of customers (excluding depot)
        customers = np.random.permutation(range(1, n))
        current_load = 0
        route = [0]  # Start at depot

        for customer in customers:
            if current_load + demands[customer] <= capacity:
                route.append(customer)
                current_load += demands[customer]
            else:
                route.append(0)  # Return to depot
                route.append(customer)
                current_load = demands[customer]

        # Close the route by returning to depot
        route.append(0)

        # Count edges and accumulate distances
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            edge_counts[(u, v)] += 1
            total_distance[(u, v)] += distance_matrix[u, v]

    # Create the heuristic matrix with adaptive weights
    heuristics_matrix = np.zeros_like(distance_matrix)
    for (u, v), count in edge_counts.items():
        if count > 0:
            avg_distance = total_distance[(u, v)] / count
            heuristics_matrix[u, v] = count / (avg_distance + 1e-6)  # Avoid division by zero

    return heuristics_matrix
