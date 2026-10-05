import copy
import logging
from .search_budget import BudgetExhausted
import random

import numpy as np
import time

from .debate import Debate
from .evolution import Evolution
import warnings
from joblib import Parallel, delayed
import re
import concurrent.futures


class InterfaceEC():
    def __init__(self, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select, n_p, timeout, use_numba,
                 **kwargs):

        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs

        self.budget = kwargs.get("budget")
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts

        # ---- Evolution (luôn dùng) ----
        self.evol = Evolution(
            api_endpoint, api_key, llm_model,
            debug_mode, prompts, **kwargs
        )

        # ---- Debate (chỉ tạo khi có advisor_configs) ----
        if kwargs.get("advisor_configs"):
            self.debate = Debate(
                api_endpoint, api_key, llm_model,
                debug_mode, prompts, **kwargs
            )
        else:
            self.debate = None
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p

        self.timeout = timeout
        self.use_numba = use_numba

    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            # Write the code to the file
            file.write(code)
        return

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate_obj(self, population, obj):
        for ind in population:
            if obj == ind['objective']:
                return True
        return False

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation_seed(self, seeds):

        population = []

        fitness = self.interface_eval.batch_evaluate([seed['code'] for seed in seeds])

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def _get_alg(self, pop, operator, advice=None, father=None):
        offspring = {
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'], offspring['thought']] = self.evol.i1(advice=advice)
        elif operator == "e1":
            real_m = random.randint(2, self.m)
            real_m = min(real_m, len(pop))
            parents = self.select.parent_selection_e1(pop, real_m)
            [offspring['code'], offspring['thought']] = self.evol.e1(parents, advice=advice)
        elif operator == "e2":
            other = copy.deepcopy(pop)
            other = [individual for individual in other if individual['code'] != father['code']]
            real_m = 1
            # real_m = random.randint(2, self.m) - 1
            # real_m = min(real_m, len(other))
            parents = self.select.parent_selection(other, real_m)
            parents.append(father)
            [offspring['code'], offspring['thought']] = self.evol.e2(parents, advice=advice)
        elif operator == "m1":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m1(parents[0], advice=advice)
        elif operator == "m2":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m2(parents[0], advice=advice)
        elif operator == "s1":
            parents = pop
            [offspring['code'], offspring['thought']] = self.evol.s1(pop, advice=advice)
        elif operator == "counter":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.counter(parents[0], advice=advice)
        elif operator == "refine":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.refine(father)

        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        offspring['algorithm'] = self.evol.post_thought(offspring['code'], offspring['thought'], advice=advice)
        return parents, offspring

    def get_offspring(self, pop, operator, advice=None, father=None):
        # Bound malformed-response retries as well as API retries.
        for attempt in range(3):
            if self.budget is not None:
                self.budget.check()
            try:
                parents, offspring = self._get_alg(pop, operator, advice=advice, father=father)
                if not self.check_duplicate(pop, offspring['code']) or attempt == 2:
                    return parents, offspring
            except BudgetExhausted:
                raise
            except Exception:
                if attempt == 2:
                    raise
                logging.exception("Candidate generation failed; retrying.")
        raise RuntimeError("Candidate generation exhausted its retries.")

    def evaluate_offspring(self, eval_times, offspring):
        """Count every evaluator invocation, including invalid outputs and AGRE."""
        if self.budget is not None:
            timeout = self.budget.remaining_seconds()
            eval_times = self.budget.reserve_evaluation()
            objs = self.interface_eval.batch_evaluate([offspring['code']], 0, timeout=timeout)
        else:
            eval_times += 1
            objs = self.interface_eval.batch_evaluate([offspring['code']], 0)
        if isinstance(objs, str) or not objs or not np.isfinite(objs[0]):
            return eval_times, None
        offspring['objective'] = float(np.round(objs[0], 5))
        return eval_times, offspring

    def get_algorithm(self, eval_times, pop, operator, advice=None):
        while True:
            if self.budget is not None:
                self.budget.check()
            _, offspring = self.get_offspring(pop, operator, advice=advice)
            eval_times, offspring = self.evaluate_offspring(eval_times, offspring)
            if offspring is not None:
                if self.budget is None and self.check_duplicate_obj(pop, offspring['objective']):
                    continue
                return eval_times, pop, offspring

    def evolve_algorithm(self, eval_times, pop, node, brother_node, operator, advice=None):
        for _ in range(3):
            if self.budget is not None:
                self.budget.check()
            _, offspring = self.get_offspring(pop, operator, advice=advice, father=node)
            eval_times, offspring = self.evaluate_offspring(eval_times, offspring)
            if offspring is not None:
                if self.budget is None and self.check_duplicate(pop, offspring['code']):
                    continue
                return eval_times, offspring
        return eval_times, None
