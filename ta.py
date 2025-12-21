import json
import os
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import numpy as np

# ====== HÀM TÌM FILE ======
def find_json_files(directory_path):
    json_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith("best_population_generation_") and file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

def get_fe_from_filename(filename):
    m = re.search(r"best_population_generation_(\d+)\.json", filename)
    return int(m.group(1)) if m else None

# ====== ĐỌC DỮ LIỆU CHO 1 RUN ======
def load_data(json_files):
    data_by_fe = {}
    for f in json_files:
        fe = get_fe_from_filename(os.path.basename(f))
        if fe is None:
            continue
        try:
            with open(f, "r") as jf:
                obj = json.load(jf).get("objective", None)
                if obj is not None:
                    data_by_fe[fe] = obj
        except:
            pass
    return data_by_fe

# ====== CẤU HÌNH 3 RUN CỦA MỖI FRAMEWORK ======
log_groups = {
    "AB-MCTS-AHD": [
        "D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/ab-mcts-ahd/2025-11-25_08-30-31",
        "D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/ab-mcts-ahd/2025-11-25_06-56-26",
        "D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/ab-mcts-ahd/2025-11-22_00-10-28",
    ],
    "MCTS-AHD": [
        "D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/mcts-ahd/2025-11-24_08-35-33",
        "D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/mcts-ahd/2025-11-24_19-07-47",
        "D:/MCTS-AHD-master/outputs/tsp_constructive-constructive/mcts-ahd/2025-11-24_20-55-31",
    ]
}

color_map = {
    "AB-MCTS-AHD": "red",
    "MCTS-AHD": "blue"
}

START_OBJ = 6.61  # Y-value where both curves start from FE=0

# ====== PLOT ======
plt.figure(figsize=(11, 5))

for label, run_paths in log_groups.items():

    # Gom dữ liệu từ 3 run
    fe_to_objs = defaultdict(list)

    for run_path in run_paths:
        json_files = find_json_files(run_path)
        data = load_data(json_files)

        for fe, obj in data.items():
            fe_to_objs[fe].append(obj)

    # Sort theo FE
    all_fes = sorted(fe_to_objs.keys())

    mean_vals = []
    std_vals = []

    for fe in all_fes:
        vals = fe_to_objs[fe]
        mean_vals.append(np.mean(vals))
        std_vals.append(np.std(vals))

    # Thêm điểm xuất phát FE = 0, obj = 6.61
    all_fes = [0] + all_fes
    mean_vals = [START_OBJ] + mean_vals
    std_vals = [0] + std_vals

    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)

    color = color_map[label]

    # Plot mean
    plt.plot(all_fes, mean_vals, color=color, label=label, linewidth=1.5)

    # Plot ± std
    plt.fill_between(all_fes, mean_vals - std_vals, mean_vals + std_vals,
                     color=color, alpha=0.15)

# ====== CUSTOM STYLE ======
plt.xlabel("Function Evaluations", fontsize=10)
plt.ylabel("Objective", fontsize=10)
plt.title("Mean ± Std of Objective Across 3 Runs", fontsize=12)
plt.legend(fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.show()
