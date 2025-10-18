import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# --- Đường dẫn tới thư mục ---
folders = [
    r"D:\MCTS-AHD-master\outputs\tsp_constructive-constructive\mcts-ahd\2025-09-23_01-06-18",
    r"D:\MCTS-AHD-master\outputs\tsp_constructive-constructive\ab-mcts-ahd\2025-10-17_00-06-40"
]
labels = ["MCTS-AHD", "AB-MCTS-AHD"]


# --- Hàm đọc file JSON ---
def read_objective_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Giả sử trong JSON có key "objective"
    return data.get("objective", None)


# --- Thu thập dữ liệu ---
gen_start, gen_end = 0, 1000
results = {}

for folder, label in zip(folders, labels):
    gen_nums = []
    obj_vals = []
    pattern = os.path.join(folder, "best_population_generation_*.json")
    files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))

    for file in files:
        gen_num = int(os.path.basename(file).split("_")[-1].split(".")[0])
        if gen_start <= gen_num <= gen_end:
            obj = read_objective_from_json(file)
            if obj is not None:
                gen_nums.append(gen_num)
                obj_vals.append(obj)

    results[label] = pd.DataFrame({"generation": gen_nums, "objective": obj_vals})

# --- Vẽ đồ thị ---
plt.figure(figsize=(12, 6))
for label, df in results.items():
    plt.plot(df["generation"], df["objective"], label=label)

plt.xlabel("Generation")
plt.ylabel("Objective")
plt.title("So sánh Objective giữa MCTS-AHD và AB-MCTS-AHD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
