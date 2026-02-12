import re

log_path = r"D:\AdaptiveMCTSAHD\outputs\cvrp_aco-aco\mcts-ahd\2025-11-28_22-24-09\main.log"

values = []

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"Objective value:\s*([0-9.+-eE]+)", line)
        if match:
            values.append(round(float(match.group(1)), 3))

print(values)
