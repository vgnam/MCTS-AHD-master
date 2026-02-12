from aco import ACO
import numpy as np
import logging
from gen_inst import OPInstance, load_dataset
import torch
import sys

sys.path.insert(0, "../../../")
import sys

sys.path.insert(0, "../../../")
sys.path.insert(0, "/kaggle/working/MCTS-AHD")
import gpt
from utils.utils import get_heuristic_name

possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)

N_ITERATIONS = 50
N_ANTS = 20


def solve(inst: OPInstance):
    heu = heuristics(np.array(inst.prize), np.array(inst.distance), inst.maxlen) + 1e-9
    assert tuple(heu.shape) == (inst.n, inst.n)
    heu[heu < 1e-9] = 1e-9
    heu = torch.from_numpy(heu)
    aco = ACO(inst.prize, inst.distance, inst.maxlen, heu, N_ANTS)
    obj, _ = aco.run(N_ITERATIONS)
    return obj


if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")



    basepath = os.path.dirname(__file__)
    for problem_size in [50, 100, 200]:
        dataset_path = os.path.join(basepath, f"dataset/val{problem_size}_dataset.npz")
        dataset = load_dataset(dataset_path)
        logging.info(f"[*] Evaluating {dataset_path}")

        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance)
            objs.append(obj.item())

        print(f"[*] Average for {problem_size}: {np.mean(objs)}")