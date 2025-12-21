import os
from ortools.algorithms.python import knapsack_solver
from os import path
import numpy as np

# Thư mục chứa dataset
base_path = r"D:\MCTS-AHD-master\problems\kp_constructive\dataset"

# Danh sách file cần chạy
datasets = [

    "test500_dataset.npy",
    "test1000_dataset.npy",
]

factor = 10000000

# Solver
solver = knapsack_solver.KnapsackSolver(
    knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    "KnapsackSolver"
)

for filename in datasets:
    file_path = path.join(base_path, filename)
    data = np.load(file_path)

    print(f"\n======================")
    print(f" Running dataset: {filename}")
    print(f"======================")

    n_instances = data.shape[0]
    total_value = 0.0

    for i in range(n_instances):
        # Knapsack capacity
        capacities = [int(25 * factor)]

        # weight = first column, value = second column
        weight = [(factor * data[i][:, 0]).astype(int).tolist()]
        value = (factor * data[i][:, 1]).astype(int).tolist()

        solver.init(value, weight, capacities)
        computed_value = solver.solve()

        print(f"Instance {i:4d} → Value = {computed_value / factor:.6f}")

        total_value += computed_value

    print(f"\n>>> Average value for {filename}: {total_value / n_instances / factor:.6f}")
