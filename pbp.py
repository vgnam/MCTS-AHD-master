import os, re
import numpy as np
import pandas as pd

# --- Roots ---
root_ab = r"D:\MCTS-AHD-master\outputs\bpp_offline_aco-aco\ab-mcts-ahd"
root_mcts = r"D:\MCTS-AHD-master\outputs\bpp_offline_aco-aco\mcts-ahd"

folders_ab = [
    "2025-10-26_00-36-32",
    "2025-10-26_08-11-08",
    "2025-10-26_17-46-34",
]
folders_mcts = [
    "2025-10-27_23-09-01",
    "2025-10-28_22-35-07",
    "2025-10-29_22-57-56",
]

instance_sizes = [120, 500, 1000]


def read_best_vals(root, folders, label):
    """Đọc từng file best_code_overall_val_stdout.txt (format: [*] Average for <size>: <float>)"""
    all_data = {size: [] for size in instance_sizes}
    detailed_rows = []  # lưu chi tiết từng lần chạy

    print(f"\n===== {label} =====")
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
                detailed_rows.append({
                    "Algorithm": label,
                    "Run folder": folder,
                    "Instance size": size,
                    "Value": val
                })
                print(f"Instance {size}: {val}")
                found_any = True

        if not found_any:
            print("(No valid results found in file)")
    return all_data, detailed_rows


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


# --- Đọc dữ liệu ---
data_ab, detail_ab = read_best_vals(root_ab, folders_ab, "AB-MCTS")
data_mcts, detail_mcts = read_best_vals(root_mcts, folders_mcts, "MCTS")

# --- Gộp dữ liệu chi tiết ---
df_detail = pd.DataFrame(detail_ab + detail_mcts)
df_detail.sort_values(by=["Algorithm", "Run folder", "Instance size"], inplace=True)

# --- Tính trung bình ---
stats_ab = summarize(data_ab)
stats_mcts = summarize(data_mcts)

df_summary = pd.DataFrame({
    "Instance size": instance_sizes,
    "AB-MCTS (mean ± std)": [stats_ab[s] for s in instance_sizes],
    "MCTS (mean ± std)": [stats_mcts[s] for s in instance_sizes],
})

# --- Xuất file ---
summary_path = "MCTS_ABMCTS_summary.csv"
detail_path = "MCTS_ABMCTS_detailed.csv"

df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

print("\n================ SUMMARY ================")
print(df_summary.to_string(index=False))
print(f"\n✅ Saved summary to: {os.path.abspath(summary_path)}")
print(f"✅ Saved detailed results to: {os.path.abspath(detail_path)}")
