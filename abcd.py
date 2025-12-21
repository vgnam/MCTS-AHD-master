import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Thay 2 đường dẫn và nhãn tương ứng nếu cần
# log_files = [
#     ("D:/MCTS-AHD-master/outputs/bpp_offline_aco-aco/ab-mcts-ahd/2025-10-26_17-46-34/main.log", "MCTS-AHD"),
#     ("D:/MCTS-AHD-master/outputs/bpp_offline_aco-aco/mcts-ahd/2025-10-29_22-57-56/main.log", "AB-MCTS-AHD"),
# ]




def parse_log(path):
    data = []
    current_tokens = None
    with open(path, "r", encoding="utf-8") as f:
        current_prompt = None
        current_completion = None
        for line in f:
            # Tìm prompt_tokens và completion_tokens
            prompt_match = re.search(r"prompt_tokens\s*=\s*(\d+)", line)
            completion_match = re.search(r"completion_tokens\s*=\s*(\d+)", line)

            if prompt_match:
                current_prompt = int(prompt_match.group(1))
            if completion_match:
                current_completion = int(completion_match.group(1))

            # Nếu có cả hai -> tính total
            if current_prompt is not None and current_completion is not None:
                current_tokens = current_prompt + current_completion

            # Tìm objective value
            objective_match = re.search(r"Objective value:\s*([0-9\.]+)", line)
            if objective_match and current_tokens is not None:
                data.append((current_tokens, float(objective_match.group(1))))

    return data

series = []
for path, label in log_files:
    pts = parse_log(path)
    if not pts:
        print(f"No valid data in {path}")
        continue
    xs, ys = zip(*pts)
    best = []
    cur_best = float('inf')
    for y in ys:
        cur_best = min(cur_best, y)
        best.append(cur_best)
    series.append((xs, best, label))

if not series:
    print("Không tìm thấy dữ liệu hợp lệ trong các file log.")
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    for xs, best, label in series:
        if label == "AB-MCTS-AHD":
            color = "tab:orange"  # màu cam
        else:
            color = None  # matplotlib tự chọn màu
        ax.plot(xs, best, linewidth=2, markersize=4, label=label, color=color)

    ax.set_xlabel("Prompt Tokens")
    ax.set_ylabel("Best Objective Value")
    ax.set_title("Best Objective Value vs Total Tokens")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x / 1000)}k"))
    plt.tight_layout()
    plt.show()
# ...existing code...