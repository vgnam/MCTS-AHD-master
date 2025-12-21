import os
import re

# --- Roots ---
root_ab = r"D:\MCTS-AHD-master\outputs\tsp_constructive-constructive\ab-mcts-ahd"
root_mcts = r"D:\MCTS-AHD-master\outputs\tsp_constructive-constructive\mcts-ahd"

# --- Folder lists ---
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

# --- Regex pattern ---
pattern = re.compile(r"LLM usage:.*?prompt_tokens\s*=\s*(\d+),\s*completion_tokens\s*=\s*(\d+)")

def extract_usage(folder_root, folder_list, tag):
    print(f"\n--- {tag} ---")
    for f in folder_list:
        log_path = os.path.join(folder_root, f, "main.log")
        if not os.path.exists(log_path):
            print(f"{f}: ❌ main.log not found")
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = [line.strip() for line in file if "LLM usage:" in line]
                if not lines:
                    print(f"{f}: ⚠️ No 'LLM usage' found")
                    continue
                last_line = lines[-1]
                match = pattern.search(last_line)
                if match:
                    prompt, completion = match.groups()
                    print(f"{f}: ✅ prompt={prompt}, completion={completion}")
                else:
                    print(f"{f}: ⚠️ Pattern not matched")
        except Exception as e:
            print(f"{f}: ❗ Error reading file - {e}")

# --- Run ---
extract_usage(root_ab, folders_ab, "AB-MCTS-AHD")
extract_usage(root_mcts, folders_mcts, "MCTS-AHD")
