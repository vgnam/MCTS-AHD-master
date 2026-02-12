# Module Name: BGEvaluation
# Last Revision: 2025/2/10
# Description: Implements a mathematical function skeleton that models the growth rate of
# E. Coli bacteria. The function takes as input observations of population density, substrate
# concentration, temperature, and pH level, along with a set of parameters to be optimized.
# The output is the calculated bacterial growth rate. This module is part of the LLM4AD project
# (https://github.com/Optima-CityU/llm4ad  ).
#
# Parameters:
# - b: np.ndarray - population density of the bacterial species (default: None).
# - s: np.ndarray - substrate concentration (default: None).
# - temp: np.ndarray - temperature observations (default: None).
# - pH: np.ndarray - pH level observations (default: None).
# - params: np.ndarray - array of numeric constants or parameters to be optimized (default: None).
# - timeout_seconds: int - Maximum allowed time (in seconds) for the evaluation process (default: 20).
#
# References:
# - Shojaee, Parshin, et al. "Llm-sr: Scientific equation discovery via programming
#   with large language models." arXiv preprint arXiv:2404.18400 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import logging
import sys

sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name

possible_func_names = ["equation", "equation_v1", "equation_v2", "equation_v3"]

equation_name = get_heuristic_name(gpt, possible_func_names)
equation = getattr(gpt, equation_name)

MAX_NPARAMS = 10

params = np.array([1.0] * MAX_NPARAMS)

def solve(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, y_true: np.ndarray):
    from scipy.optimize import minimize

    def loss(params):
        y_pred = equation(b, s, temp, pH, params)
        return np.mean((y_pred - y_true) ** 2)

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')  # FIX: Sử dụng list mới, không dùng biến global params

    loss_value = result.fun
    if np.isnan(loss_value) or np.isinf(loss_value):
        return None
    else:
        return 10 * loss_value


if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    # mood = 'train'

    basepath = os.path.dirname(__file__)

    if mood == 'train':
        import train

        df = pd.DataFrame(train.data)
        data = np.array(df)

        X = data[:, :-1]
        y = data[:, -1]  # FIX: Trích xuất ground truth labels
        n_instances = 1

        print(f"[*] Dataset loaded with {n_instances} instances.")

        objs = []
        b, s, temp, pH = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        obj = solve(b, s, temp, pH, y)  # FIX: Truyền y thay vì params
        print(f"[*] Instance 0: {obj}")
        objs.append(obj)

        print("[*] Average:")
        print(np.mean(objs))

    else:  # mood == 'val'
        import test2

        df = pd.DataFrame(test2.data)
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1]  # FIX: Trích xuất ground truth labels
        logging.info(f"[*] Evaluating validation dataset")

        objs = []
        b, s, temp, pH = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        obj = solve(b, s, temp, pH, y)  # FIX: Truyền y thay vì params
        objs.append(obj)

        print(f"[*] Average: {np.mean(objs)}")