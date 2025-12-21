# import json
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import re
#
# # --- Paths ---
# # log_files = [
# #     ("D:/MCTS-AHD-master/outputs/bpp_offline_aco-aco/ab-mcts-ahd/2025-10-26_17-46-34", "AB-MCTS-AHD"),
# #     ("D:/MCTS-AHD-master/outputs/bpp_offline_aco-aco/mcts-ahd/2025-10-29_22-57-56", "MCTS-AHD"),
# # ]
#
# log_files = [
#      ("D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/ab-mcts-ahd/2025-11-25_08-30-31", "AB-MCTS-AHD"),
#      ("D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/mcts-ahd/2025-11-24_08-35-33", "MCTS-AHD"),
# ]
#
# # Fixed color map
# color_map = {
#     "AB-MCTS-AHD": "red",
#     "MCTS-AHD": "blue"
# }
#
# # --- 1. Find all population_generation_XX.json files ---
# def find_population_files(directory_path):
#     results = []
#     for root, dirs, files in os.walk(directory_path):
#         for file in files:
#             if file.startswith("population_generation_") and file.endswith(".json"):
#                 results.append(os.path.join(root, file))
#     return results
#
#
# # --- 2. Extract generation index from filename ---
# def get_gen_from_filename(filename):
#     match = re.search(r"population_generation_(\d+)\.json", filename)
#     return int(match.group(1)) if match else None
#
#
# # --- 3. Extract objectives grouped by generation ---
# def extract_generation_data(json_files):
#     gen_to_obj = {}
#
#     for f in json_files:
#         gen = get_gen_from_filename(os.path.basename(f))
#         if gen is None:
#             continue
#
#         try:
#             with open(f, "r") as jf:
#                 data = json.load(jf)
#
#                 # Expecting a list of objects like your example
#                 if isinstance(data, list):
#                     for entry in data:
#                         obj = entry.get("objective", None)
#                         if obj is not None:
#                             gen_to_obj.setdefault(gen, []).append(obj)
#
#         except Exception as e:
#             print(f"Warning loading {f}: {e}")
#
#     # Sort by generation index
#     gens = sorted(gen_to_obj.keys())
#     obj_arrays = [np.array(gen_to_obj[g]) for g in gens]
#
#     means = [arr.mean() for arr in obj_arrays]
#     stds = [arr.std() for arr in obj_arrays]
#
#     return gens, means, stds
#
#
# # --- 4. Plot with variance shade ---
# plt.figure(figsize=(11, 5))
#
# for path, label in log_files:
#     json_files = find_population_files(path)
#     print(f"{label}: Found {len(json_files)} population JSONs")
#
#     gens, means, stds = extract_generation_data(json_files)
#
#     if len(gens) == 0:
#         print(f"⚠ No valid data found for {label}")
#         continue
#
#     gens = np.array(gens)
#     means = np.array(means)
#     stds = np.array(stds)
#
#     color = color_map[label]
#
#     # Mean line
#     plt.plot(gens, means, color=color, label=label, linewidth=1.5)
#
#     # Shade for std
#     plt.fill_between(gens, means - stds, means + stds, color=color, alpha=0.2)
#
#
# plt.xlabel("Generation")
# plt.ylabel("Objective")
# plt.title("Mean ± Std Objective per Generation")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.ylim(bottom=np.min(means) * 0.5, top=np.max(means) * 1.5)
#
# plt.show()

import json
import os
import matplotlib.pyplot as plt
import re

# Log folders (pointing to the main run directories, not the log files themselves)
log_files = [
     ("D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/ab-mcts-ahd/2025-11-25_08-30-31", "AB-MCTS-AHD"),
    # 2025-11-25_06-56-26
    # 2025-11-22_00-10-28
     ("D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/mcts-ahd/2025-11-24_08-35-33", "MCTS-AHD"),
    # 2025-11-24_19-07-47
    # 2025-11-24_20-55-31
]

# Fixed colors
color_map = {
    "AB-MCTS-AHD": "red",
    "MCTS-AHD": "blue"
}

# Find JSON files in folder (recursive search)
def find_json_files(directory_path):
    """
    Recursively searches for files matching the pattern best_population_generation_*.json
    within the given directory_path.
    """
    json_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith("best_population_generation_") and file.endswith(".json"):
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    return json_files

# Extract FE from filename
def get_fe_from_filename(filename):
    match = re.search(r"best_population_generation_(\d+)\.json", filename)
    if match:
        return int(match.group(1))
    else:
        return None

# Extract objective and FE
def extract_data(json_files):
    fe_list = []
    obj_list = []
    for f in json_files:
        fe = get_fe_from_filename(os.path.basename(f))
        if fe is None:
            continue
        try:
            with open(f, "r") as jf:
                data = json.load(jf)
                obj = data.get("objective", None)
                if obj is not None:
                    fe_list.append(fe)
                    obj_list.append(obj)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from file {f}. Skipping.")
        except FileNotFoundError:
            print(f"Warning: File {f} not found during processing. Skipping.")
    # Sort by FE
    if fe_list and obj_list:
        fe_obj = sorted(zip(fe_list, obj_list))
        fe_list, obj_list = zip(*fe_obj)
    return list(fe_list), list(obj_list)

START_OBJ = 6.61

plt.figure(figsize=(10, 5))

found_any_data = False
for log_path, label in log_files:
    json_files = find_json_files(log_path)
    print(f"Found {len(json_files)} matching JSON files in {label}: {json_files[:3]}...")

    fe, obj = extract_data(json_files)

    if fe and obj:
        # Thêm điểm xuất phát FE = 0 và Objective = 6.61
        fe = [0] + fe
        obj = [START_OBJ] + obj

        plt.plot(fe, obj, color=color_map[label], label=label, linewidth=1)
        found_any_data = True
    else:
        print(f"Warning: No valid FE/Objective data found in {label}.")

if found_any_data:
    plt.xlabel("Function Evaluations", fontsize=10, fontweight='light')     # chữ mỏng
    plt.ylabel("Objective", fontsize=10, fontweight='light')
    plt.title("Objective vs FE for 2 frameworks", fontsize=12, fontweight='light')
    plt.grid(False)
    plt.legend(fontsize=10, frameon=False)  # legend chữ nhỏ, mỏng
    plt.tight_layout()
    plt.show()
else:
    print("No data found to plot for any of the specified directories.")
