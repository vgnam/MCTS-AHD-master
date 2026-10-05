from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

from .evolution_interface import InterfaceEC
from .ab_mcts import MCTSNode, AB_MCTS_A, AblationConfig
from .search_budget import BudgetExhausted, SearchBudget


class AB_MCTS_A_AHD:
    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage
        self.paras = paras
        self.config = kwargs.get('cfg')

        # LLM settings - support single or multiple LLMs
        self.llm_model_names = []  # List of LLM model names

        # Handle single or multiple LLM model names
        if hasattr(paras, 'llm_model_names') and paras.llm_model_names:
            if isinstance(paras.llm_model_names, list):
                self.llm_model_names = paras.llm_model_names
            else:
                self.llm_model_names = [paras.llm_model_names]
        elif hasattr(paras, 'llm_model') and paras.llm_model:
            self.llm_model_names = [paras.llm_model]  # Fixed: wrap in list
        else:
            # Fallback to default model
            self.llm_model_names = ['default_model']

        # Store LLM configurations
        self.llm_configs = {}
        if hasattr(paras, 'llm_configs') and paras.llm_configs:
            self.llm_configs = paras.llm_configs
        else:
            # Create default config for each model
            for model_name in self.llm_model_names:
                self.llm_configs[model_name] = {
                    'api_endpoint': getattr(paras, 'llm_api_endpoint', ''),
                    'api_key': getattr(paras, 'llm_api_key', ''),
                    'use_local': kwargs.get('use_local_llm', False),
                    'url': kwargs.get('url', ''),
                }

        # Experimental settings
        self.init_size = paras.init_size
        self.pop_size = paras.pop_size
        self.fe_max = paras.ec_fe_max
        self.budget = SearchBudget(self.fe_max, (self.config or {}).get("search_timeout", 60))
        self.batch_size = int((self.config or {}).get("candidates_per_expansion", 4))
        if min(self.init_size, self.pop_size, self.batch_size) <= 0:
            raise ValueError("Pool and expansion sizes must be positive.")

        self.m = 5

        self.debug_mode = paras.exp_debug_mode
        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        # Ablation settings (missing keys = full GENESIS)
        abl = (self.config.get("ablation") if self.config is not None else None) or {}
        self.ablation = AblationConfig(
            posterior_sampling=bool(abl.get("posterior_sampling", True)),
            adaptive_op_selection=bool(abl.get("adaptive_op_selection", True)),
            global_sharing=bool(abl.get("global_sharing", True)),
            ucb_c=float(abl.get("ucb_c", 1.0)),
        )
        self.use_agre = bool(abl.get("agre", True))
        self.agre_stages = dict(abl.get("agre_stages") or {})

        print("- GENESIS parameters loaded -")
        print(f"LLM Models: {self.llm_model_names}")
        print(f"Ablation: {self.ablation}, AGRE: {self.use_agre}, AGRE stages: {self.agre_stages}")
        seed = (self.config or {}).get("seed", 2024)
        random.seed(seed)
        np.random.seed(seed)


    @property
    def eval_times(self):
        return self.budget.evaluations

    def _add_candidate(self, mcts, parent, population, offspring, model, operator,
                       method="GEN"):
        # Every evaluated candidate is evidence, even if its objective/description
        # matches an existing one. Only the elite pool is deduplicated by code.
        node = MCTSNode(
            algorithm=offspring['algorithm'], code=offspring['code'],
            obj=float(offspring['objective']), parent=parent,
            depth=parent.depth + 1, raw_info=offspring,
            llm_model_names=self.llm_model_names, global_hyper=mcts.global_hyper)
        parent.add_child(node, generation_method=method, generation_action=model)
        parent.children_info.append(offspring)
        mcts.backpropagate(node, operator)
        same_code = next((i for i, item in enumerate(population)
                          if item['code'] == offspring['code']), None)
        if same_code is None:
            population.append(offspring)
        elif offspring['objective'] < population[same_code]['objective']:
            population[same_code] = offspring
        population.sort(key=lambda item: item['objective'])
        del population[self.pop_size:]
        return node

    def expand(self, mcts, cur_node, nodes_set, option, model_name=None, op_w=None):
        interface = self.interface_ecs[model_name or self.llm_model_names[0]]
        if option == 's1':
            path_set = []
            current = cur_node
            while not current.is_root:
                path_set.append(current.raw_info)
                current = current.parent
            if len(path_set) < 2:
                raise ValueError("S1 requires two real nodes on the selected path.")
        elif option == 'e1':
            path_set = [child.raw_info for child in mcts.root.children]
        else:
            path_set = nodes_set
        all_offsprings = []
        for _ in range(self.batch_size if op_w is None else op_w):
            self.budget.check()
            _, offspring = interface.evolve_algorithm(
                self.eval_times, path_set, cur_node.raw_info,
                cur_node.children_info, option)
            if offspring is not None:
                all_offsprings.append(offspring)
                self._add_candidate(mcts, cur_node, nodes_set, offspring,
                                    model_name or self.llm_model_names[0], option)

        if self.use_agre and all_offsprings:
            self.budget.check()
            best = min(all_offsprings, key=lambda item: item['objective'])
            try:
                _, refined = interface.get_offspring(path_set, "refine", father=best)
                _, refined = interface.evaluate_offspring(self.eval_times, refined)
            except BudgetExhausted:
                raise
            except Exception:
                logging.exception("AGRE refinement failed; retaining evaluated candidates.")
            else:
                if refined is not None:
                    self._add_candidate(mcts, cur_node, nodes_set, refined,
                                        model_name or self.llm_model_names[0], option,
                                        method="AGRE")
        return nodes_set

    def _save_population(self, population):
        directory = Path(self.output_path)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = f"generation_{self.eval_times}.json"
        (directory / f"population_{suffix}").write_text(
            json.dumps(population, indent=2), encoding="utf-8")
        filename = directory / f"best_population_{suffix}"
        filename.write_text(json.dumps(population[0], indent=2), encoding="utf-8")
        return str(filename)

    def run(self):
        self.budget.start()
        self.interface_ecs = {}
        for model_name in self.llm_model_names:
            config = self.llm_configs.get(model_name, {})
            self.interface_ecs[model_name] = InterfaceEC(
                self.m, config.get('api_endpoint', ''), config.get('api_key', ''),
                model_name, self.debug_mode, self.prob, select=self.select,
                n_p=self.exp_n_proc, timeout=self.timeout, use_numba=self.use_numba,
                use_local_llm=config.get('use_local', False), url=config.get('url', ''),
                agre_stages=self.agre_stages, budget=self.budget)
        self.mcts = AB_MCTS_A('Root', self.llm_model_names, ablation=self.ablation)
        population = []
        try:
            # N0 is the total pool size, including when multiple models are used.
            i = 0
            while len(self.mcts.root.children) < self.init_size:
                self.budget.check()
                model = self.llm_model_names[i % len(self.llm_model_names)]
                i += 1
                operator = 'i1' if not population else 'e1'
                _, _, offspring = self.interface_ecs[model].get_algorithm(
                    self.eval_times, population, operator)
                if offspring is not None:
                    self._add_candidate(self.mcts, self.mcts.root, population,
                                        offspring, model, operator)
                    self._save_population(population)
            if not population:
                raise RuntimeError("Initialization produced no valid heuristic.")
            while True:
                self.budget.check()
                target, _, (model, operator) = self.mcts.select_expansion_target(
                    self.eval_times, population=population)
                self.expand(self.mcts, target, population, operator, model)
                self._save_population(population)
        except BudgetExhausted as exc:
            logging.info("Search stopped after %s evaluations: %s", self.eval_times, exc)
        if not population:
            raise RuntimeError("Search budget exhausted before any valid heuristic was evaluated.")
        filename = self._save_population(population)
        return population[0]['code'], filename
