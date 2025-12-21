import math
from os import path
import numpy as np
import logging
import copy
from tqdm import tqdm
import time

# import GPT heuristic function
try:
    from gpt import select_next_item_v2 as select_next_item
except:
    from gpt import select_next_item

def eval_heuristic(node_positions: np.ndarray, capacity: float) -> float:
    """
    Generate solution for KP problem using GPT-generated heuristic.
    """
    weight = node_positions[:, 0].copy()
    value = node_positions[:, 1].copy()
    remaining_capacity = copy.deepcopy(capacity)
    solution_value = 0.0

    while len(weight) > 0 and remaining_capacity + 1e-6 >= min(weight):
        next_node = select_next_item(remaining_capacity=remaining_capacity, weights=weight, values=value)
        if next_node < len(value) and remaining_capacity + 1e-6 >= weight[next_node]:
            solution_value += value[next_node]
            remaining_capacity -= weight[next_node]
            weight = np.delete(weight, next_node)
            value = np.delete(value, next_node)
        else:
            break

    return solution_value

if __name__ == '__main__':
    datasets = [50, 100, 200, 500, 1000]

    # Replace these with true optimal values per instance if available
    optimal_values = {
        50: [
            24.687278, 25.198997, 25.557301, 23.504142, 26.725183, 24.372912,
            27.037336, 24.187221, 26.241273, 23.588902, 23.303182, 28.240942,
            28.509215, 23.161434, 26.595432, 22.407652, 25.760580, 24.407697,
            23.244944, 22.874410, 22.959593, 23.266560, 24.940675, 25.510562,
            24.892496, 27.638919, 24.941451, 20.147273, 26.356271, 24.682694,
            26.900267, 26.286949, 26.483747, 17.602575, 24.369346, 25.649747,
            29.101189, 20.650239, 24.384778, 24.384381, 28.277117, 25.602133,
            25.275768, 25.121157, 25.073425, 29.682340, 21.694668, 24.175502,
            24.031665, 22.909876, 26.629715, 27.179320, 22.091206, 22.235990,
            20.809175, 26.856935, 27.886852, 23.885046, 24.828745, 25.025783,
            21.577271, 26.519217, 30.732693, 23.703731
        ],
        100: [
            39.398690, 45.584036, 44.732243, 37.940095, 39.835651, 38.672960,
            38.466162, 39.297468, 42.330312, 40.406242, 44.798224, 38.465693,
            41.355227, 42.621685, 40.125512, 41.339719, 41.022819, 42.316716,
            39.971180, 41.137064, 38.168338, 40.573402, 44.367096, 41.191883,
            38.550604, 44.336969, 42.931044, 42.628580, 41.810364, 40.293898,
            38.838114, 35.833405, 43.310244, 39.129063, 38.863033, 42.635338,
            45.159517, 41.572383, 40.301865, 40.613851, 45.329498, 39.732835,
            43.513411, 37.811083, 39.221790, 38.847679, 45.444037, 42.092938,
            37.571568, 36.151065, 42.599541, 36.518475, 39.910694, 41.275026,
            35.606048, 37.481791, 47.213886, 38.242360, 34.990291, 39.745304,
            40.655974, 39.308504, 39.105141, 45.149857
        ],
        200: [
            53.144796, 59.695471, 61.680864, 63.598056, 60.409984, 54.494538,
            54.395013, 61.184718, 52.994106, 57.135402, 56.877376, 58.829882,
            56.686117, 55.221327, 63.015898, 60.004777, 56.646453, 60.956232,
            58.544236, 57.447613, 56.948782, 53.894543, 59.748104, 62.857343,
            58.998892, 55.547857, 55.717039, 62.120988, 54.144318, 61.205394,
            60.713661, 60.996009, 56.913412, 54.415077, 55.873305, 55.148654,
            55.344950, 58.448891, 54.580632, 66.499915, 63.657270, 59.801500,
            59.477960, 60.676054, 56.378078, 59.000265, 56.415241, 52.391133,
            60.493855, 53.430552, 58.326079, 53.563073, 55.391742, 57.414115,
            59.714395, 57.946762, 58.880767, 54.378272, 57.631740, 57.700632,
            55.246460, 54.898553, 62.609977, 56.315318
        ]
    }

    optimal_values.update({
        500: [
            85.723162, 87.135436, 85.624247, 101.718996, 88.385303, 88.657036,
            83.562648, 86.960782, 87.173182, 94.341141, 90.536181, 89.471411,
            98.945042, 85.904728, 85.253343, 90.523476, 98.423528, 88.613311,
            93.155858, 96.414311, 92.329890, 92.662621, 91.176038, 93.979794,
            92.658016, 90.061339, 94.968824, 91.694981, 94.723506, 81.691630,
            90.356307, 89.879620, 93.766094, 96.153419, 91.886807, 87.457343,
            92.187190, 91.480314, 93.255419, 93.102368, 84.266034, 98.109959,
            86.917848, 94.321969, 99.754206, 84.597906, 87.062031, 96.722870,
            101.745105, 89.125317, 89.644199, 88.833111, 91.283338, 88.097520,
            89.235449, 88.824657, 87.344544, 96.174064, 89.467967, 79.332807,
            90.144004, 92.240344, 97.290071, 91.750996
        ],
        1000: [
            127.306920, 121.483996, 129.636820, 115.412022, 132.009173, 134.826053,
            134.498058, 128.459022, 132.908926, 125.304476, 131.137069, 118.045821,
            125.263373, 125.092706, 125.823994, 130.075284, 137.497513, 125.955325,
            131.286390, 127.305825, 133.231199, 122.885460, 131.584690, 121.153424,
            137.901202, 126.302227, 128.366038, 131.417771, 139.667272, 125.108331,
            132.713997, 120.040104, 135.005859, 129.207533, 122.120370, 130.830705,
            125.984912, 125.881670, 132.892033, 133.732047, 131.811042, 124.401513,
            123.247981, 132.905777, 122.521414, 128.720960, 128.843682, 124.267870,
            126.398138, 125.133566, 127.333539, 125.244717, 119.586866, 126.204271,
            134.311865, 132.771756, 124.502708, 125.457576, 134.825512, 140.155215,
            125.513156, 133.666479, 124.963466, 119.438186
        ]
    })

    for problem_size in datasets:
        # For 50, 100, 200 use 'val', for 500, 1000 use 'test'
        prefix = "val" if problem_size in [50, 100, 200] else "test"
        dataset_path = path.join("dataset", f"{prefix}{problem_size}_dataset.npy")

        capacity = 25

        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]

        objs = []
        gaps = []
        runtimes = []

        for i in range(n_instances):
            start_time = time.time()
            obj = eval_heuristic(node_positions[i], capacity)
            runtime = time.time() - start_time

            objs.append(obj)
            runtimes.append(runtime)

            # Calculate gap if optimal values are provided
            if optimal_values[problem_size] is not None:
                gap = (- obj + optimal_values[problem_size][i]) / optimal_values[problem_size][i] * 100
                gaps.append(gap)

        print(f"\n[*] Summary for size {problem_size}:")
        print(f"Average value: {np.mean(objs):.6f}")
        if gaps:
            print(f"Average gap (%): {np.mean(gaps):.6f}")
        print(f"Average runtime (s): {np.mean(runtimes):.6f}")

        # # Optional: print table per instance
        # print("\nInstance | Value      | Runtime (s) | Gap (%)")
        # print("---------------------------------------------")
        # for i in range(n_instances):
        #     gap_str = f"{gaps[i]:.4f}" if gaps else "-"
        #     print(f"{i:8d} | {objs[i]:10.4f} | {runtimes[i]:10.4f} | {gap_str}")
        print("\n" + "="*50 + "\n")


# AB MCTS:


#
# [*] Summary for size 50:
# Average value: 24.880386
# Average gap (%): 0.013598
# Average runtime (s): 0.012471
#
# ==================================================
#
#
# [*] Summary for size 100:
# Average value: 40.660065
# Average gap (%): 0.084577
# Average runtime (s): 0.021950
#
# ==================================================
#
#
# [*] Summary for size 200:
# Average value: 57.835016
# Average gap (%): 0.091528
# Average runtime (s): 0.050521
#
# ==================================================
#
#
# [*] Summary for size 500:
# Average value: 90.854252
# Average gap (%): 0.165398
# Average runtime (s): 0.192450
#
# ==================================================
#
#
# [*] Summary for size 1000:
# Average value: 127.933921
# Average gap (%): 0.241800
# Average runtime (s): 0.720535


#
# [*] Summary for size 50:
# Average value: 24.880480
# Average gap (%): 0.013188
# Average runtime (s): 0.010578
#
# ==================================================
#
#
# [*] Summary for size 100:
# Average value: 40.660181
# Average gap (%): 0.083671
# Average runtime (s): 0.030503
#
# ==================================================
#
#
# [*] Summary for size 200:
# Average value: 57.835992
# Average gap (%): 0.090416
# Average runtime (s): 0.099905
#
# ==================================================
#
#
# [*] Summary for size 500:
# Average value: 90.941714
# Average gap (%): 0.068868
# Average runtime (s): 0.450999
#
# ==================================================
#
#
# [*] Summary for size 1000:
# Average value: 128.176161
# Average gap (%): 0.052511
# Average runtime (s): 1.770808


#
# [*] Summary for size 50:
# Average value: 24.880480
# Average gap (%): 0.013188
# Average runtime (s): 0.001393
#
# ==================================================
#
#
# [*] Summary for size 100:
# Average value: 40.663427
# Average gap (%): 0.074915
# Average runtime (s): 0.002127
#
# ==================================================
#
#
# [*] Summary for size 200:
# Average value: 57.840450
# Average gap (%): 0.082873
# Average runtime (s): 0.004580
#
# ==================================================
#
#
# [*] Summary for size 500:
# Average value: 90.952700
# Average gap (%): 0.057039
# Average runtime (s): 0.012218
#
# ==================================================
#
#
# [*] Summary for size 1000:
# Average value: 128.190735
# Average gap (%): 0.041198
# Average runtime (s): 0.027132

