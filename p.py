import os, re
import numpy as np
import pandas as pd

# --- Roots ---
root_ab = r"D:\MCTS-AHD-master\outputs\tsp_constructive-constructive\ab-mcts-ahd"
root_mcts = r"D:\MCTS-AHD-master\outputs\tsp_constructive-constructive\mcts-ahd"

folders_ab = [
    "2025-10-19_11-35-32",
    "2025-10-19_18-53-16",
    "2025-10-19_20-10-02",
    "2025-10-20_02-09-24",
    "2025-10-20_20-38-33",
    "2025-10-20_22-31-50",
    "2025-10-28_08-43-09",
]
folders_mcts = [
    "2025-10-21_22-21-04",
    "2025-10-21_08-18-39",
    "2025-10-20_07-49-07",
    "2025-10-20_00-47-37",
    "2025-10-19_21-20-18",
    "2025-10-19_23-28-23",
    "2025-10-12_21-50-42",
]
instance_sizes = [20, 50, 100, 200]


def read_best_vals(root, folders, label):
    """Đọc từng file best_code_overall_val_stdout.txt (format: [*] Average for <size>: <float>)"""
    all_data = {size: [] for size in instance_sizes}
    print(f"\n===== {label} =====")

    # Regex khớp đúng mẫu
    pattern = re.compile(r"\[\*\]\s*Average\s*for\s*(\d+):\s*([-+]?\d*\.\d+|\d+)")

    for folder in folders:
        file_path = os.path.join(root, folder, "best_code_overall_val_stdout.txt")
        if not os.path.exists(file_path):
            print(f"[!] Missing file in {folder}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"\n--- {label}: {folder} ---")
        found_any = False
        for match in pattern.finditer(text):
            size = int(match.group(1))
            val = float(match.group(2))
            if size in instance_sizes:
                all_data[size].append(val)
                print(f"Instance {size}: {val}")
                found_any = True

        if not found_any:
            print("(No valid results found in file)")
    return all_data


def summarize(data_dict):
    """Tính mean ± std"""
    stats = {}
    for size, vals in data_dict.items():
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            stats[size] = f"{mean:.4f} ± {std:.4f}"
        else:
            stats[size] = "N/A"
    return stats


# --- Đọc & in ---
data_ab = read_best_vals(root_ab, folders_ab, "AB-MCTS")
data_mcts = read_best_vals(root_mcts, folders_mcts, "MCTS")

# --- Tóm tắt ---
stats_ab = summarize(data_ab)
stats_mcts = summarize(data_mcts)

df_summary = pd.DataFrame({
    "Instance size": instance_sizes,
    "AB-MCTS (mean ± std)": [stats_ab[s] for s in instance_sizes],
    "MCTS (mean ± std)": [stats_mcts[s] for s in instance_sizes],
})

print("\n================ SUMMARY ================")
print(df_summary.to_string(index=False))
