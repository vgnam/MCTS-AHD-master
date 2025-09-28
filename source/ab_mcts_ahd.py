from __future__ import annotations
import random
import copy
import math
from collections import deque, defaultdict
from enum import Enum
import tqdm
import numpy as np
from scipy.stats import invgamma
import copy
import json
import random
import time
from .evolution_interface import InterfaceEC
from .ab_mcts import GENNode, MCTSNode, AB_MCTS_A


class AB_MCTS_A_AHD:
    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage

        # LLM settings - support single or multiple LLMs
        self.llm_model_names = []  # List of LLM model names

        # Handle single or multiple LLM model names
        if hasattr(paras, 'llm_model_names') and paras.llm_model_names:
            if isinstance(paras.llm_model_names, list):
                self.llm_model_names = paras.llm_model_names
            else:
                self.llm_model_names = [paras.llm_model_names]
        elif hasattr(paras, 'llm_model') and paras.llm_model:
            self.llm_model_names = paras.llm_model
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
        self.eval_times = 0

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        paras.ec_m = 5
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode
        self.ndelay = 1

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        print("- AB-MCTS-A Multiple LLM parameters loaded -")
        print(f"LLM Models: {self.llm_model_names}")
        random.seed(2024)

    def add2pop(self, population, offspring):
        """Add offspring to population, avoiding duplicates"""
        for ind in population:
            if 'algorithm' in ind and ind['algorithm'] == offspring['algorithm']:
                if self.debug_mode:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def expand(self, mcts, cur_node, nodes_set, option, model_name=None):
        """Expand using specific operator and LLM model"""
        # Get the interface for this specific model
        interface_ec = self.interface_ecs.get(model_name,
                                              list(self.interface_ecs.values())[0]) if self.interface_ecs else None

        if option == 's1':
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.code != "Root":
                path_set.append(now.raw_info)
                now = copy.deepcopy(now.parent)
            path_set = self.manage.population_management_s1(path_set, len(path_set))
            if len(path_set) == 1:
                return nodes_set
            self.eval_times, offsprings = interface_ec.evolve_algorithm(self.eval_times, path_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option)
        elif option == 'e1':
            e1_set = [copy.deepcopy(children.subtree[random.choices(range(len(children.subtree)), k=1)[0]].raw_info) for
                      children in mcts.root.children]
            self.eval_times, offsprings = interface_ec.evolve_algorithm(self.eval_times, e1_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option)
        else:
            self.eval_times, offsprings = interface_ec.evolve_algorithm(self.eval_times, nodes_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option)
        if offsprings == None:
            print(f"Timeout emerge, no expanding with action {option}.")
            return nodes_set

        if option != 'e1':
            print(
                f"Action: {option}, Father Obj: {cur_node.raw_info['objective']}, Now Obj: {offsprings['objective']}, Depth: {cur_node.depth + 1}")
        else:
            print(f"Action: {option}, Father is Root, Now Obj: {offsprings['objective']}")

        if offsprings['objective'] != float('inf'):
            success = self.add2pop(nodes_set, offsprings)
            if success:
                size_act = min(len(nodes_set), self.pop_size)
                nodes_set = self.manage.population_management(nodes_set, size_act)

                new_reward = float(offsprings['objective'])  # Reward equals objective (unbounded)
                # Create new node with reward as obj parameter
                new_node = MCTSNode(
                    algorithm=offsprings['algorithm'],
                    code=offsprings['code'],
                    obj=new_reward,  # obj parameter will be used to calculate internal reward
                    parent=cur_node,
                    depth=getattr(cur_node, 'depth', 0) + 1,
                    visit=1,
                    raw_info=offsprings,
                    llm_model_names=self.llm_model_names
                )

                # --- Key Addition: Store generation method and action ---
                # This is crucial for the backpropagate method to work correctly
                new_node._generation_method = 'GEN'  # Indicates this node was generated by GEN action
                new_node._generation_action = model_name  # The specific LLM model used
                # -------------------------------------------------------------

                cur_node.add_child(new_node)
                cur_node.children_info.append(offsprings)

                if hasattr(mcts, 'backpropagate'):
                    mcts.backpropagate(new_node)

                # Print tree rewards after successful expansion
                print(f"\n--- Tree after operator '{option}' with model '{model_name}' ---")
                self.print_tree_rewards(mcts)
                print("--- End Tree ---\n")

        return nodes_set

    def run(self):
        print("- Initialization Start -")

        # Create interface_ec for each LLM model
        self.interface_ecs = {}
        for model_name in self.llm_model_names:
            config = self.llm_configs.get(model_name, {})
            interface_ec = InterfaceEC(self.m,
                                       config.get('api_endpoint', ''),
                                       config.get('api_key', ''),
                                       model_name,
                                       self.debug_mode,
                                       self.prob,
                                       use_local_llm=config.get('use_local', False),
                                       url=config.get('url', ''),
                                       select=self.select,
                                       n_p=self.exp_n_proc,
                                       timeout=self.timeout,
                                       use_numba=self.use_numba)
            self.interface_ecs[model_name] = interface_ec
            print(f"Created interface for model: {model_name}")

        mcts = AB_MCTS_A('Root', self.llm_model_names)  # Pass LLM model names to MCTS

        # Initialize with some solutions using first LLM
        first_model = self.llm_model_names[0] if self.llm_model_names else 'default'
        first_interface = self.interface_ecs.get(first_model,
                                                 list(self.interface_ecs.values())[0]) if self.interface_ecs else None

        if first_interface:
            self.eval_times, brothers, offsprings = first_interface.get_algorithm(self.eval_times, [], "i1")
            if offsprings is not None:
                brothers = [offsprings]  # Initialize brothers list
                new_node = MCTSNode(offsprings['algorithm'], offsprings['code'], float(offsprings['objective']),
                                    parent=mcts.root, depth=1, visit=1,
                                    raw_info=offsprings,
                                    llm_model_names=self.llm_model_names)

                # --- Key Addition: Store generation method and action for root initialization ---
                new_node._generation_method = 'GEN'
                new_node._generation_action = first_model
                # ----------------------------------------------------------------------------------

                mcts.root.add_child(new_node)
                mcts.root.children_info.append(offsprings)
                mcts.backpropagate(new_node)

        # Add more initial solutions
        model_idx = 0
        for i in range(1, self.init_size):
            # Cycle through LLMs for initialization
            model_name = self.llm_model_names[
                model_idx % len(self.llm_model_names)] if self.llm_model_names else 'default'
            interface_ec = self.interface_ecs.get(model_name, list(self.interface_ecs.values())[
                0]) if self.interface_ecs else None

            if interface_ec:
                self.eval_times, brothers, offsprings = interface_ec.get_algorithm(self.eval_times,
                                                                                   brothers if 'brothers' in locals() else [],
                                                                                   "e1")
                if offsprings is not None and offsprings['objective'] != float('inf'):
                    if 'brothers' not in locals():
                        brothers = []
                    brothers.append(offsprings)
                    new_node = MCTSNode(offsprings['algorithm'], offsprings['code'], offsprings['objective'],
                                        parent=mcts.root, depth=1, visit=1,
                                        raw_info=offsprings,
                                        llm_model_names=self.llm_model_names)

                    # --- Key Addition: Store generation method and action for initialization ---
                    new_node._generation_method = 'GEN'
                    new_node._generation_action = model_name
                    # ----------------------------------------------------------------------------------

                    mcts.root.add_child(new_node)
                    mcts.root.children_info.append(offsprings)
                    mcts.backpropagate(new_node)
            model_idx += 1

        nodes_set = brothers if 'brothers' in locals() else []
        size_act = min(len(nodes_set), self.pop_size)
        nodes_set = self.manage.population_management(nodes_set, size_act)
        print("- Initialization Finished - Evolution Start -")

        while self.eval_times < self.fe_max:
            print(f"Current rewards of MCTS nodes: {[round(float(x), 2) for x in mcts.rank_list[-5:]]}")

            # AB-MCTS-A Multiple LLM selection and expansion
            target_node, expansion_type, selected_model_name = mcts.select_expansion_target()
            print(
                f"Iter: {self.eval_times}/{self.fe_max} Type: {expansion_type}, Selected Model: {selected_model_name}")

            # Apply operators according to weights
            n_op = len(self.operators)
            for i in range(n_op):
                op = self.operators[i]
                op_w = self.operator_weights[i] if i < len(self.operator_weights) else 1

                print(f"OP: {op} (weight: {op_w})", end="|")

                # Apply this operator op_w times with selected LLM
                for j in range(op_w):
                    nodes_set = self.expand(mcts, target_node, nodes_set, op, selected_model_name)

            # Population management
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)

            # Save population to a file
            filename = self.output_path + "population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set, f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "best_population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set[0]["code"], f, indent=5)

        return nodes_set[0]["code"], filename

    def print_tree_rewards(self, mcts, indent=0):

        def print_node(node, level=0):
            spaces = "  " * level
            depth = getattr(node, "depth", level)
            print(
                f"{spaces}Node: {node.code[:30] if hasattr(node, 'code') else 'Root'} "
                f"| Depth: {depth} "
                f"| Reward: {getattr(node, 'reward', 'N/A'):.2f} "
                f"| Visits: {getattr(node, 'visits', 0)}"
            )
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    print_node(child, level + 1)

        def get_max_depth(node):
            if not hasattr(node, 'children') or not node.children:
                return getattr(node, "depth", 0)
            return max(get_max_depth(child) for child in node.children)

        print("=== MCTS Tree Rewards ===")
        print_node(mcts.root)
        max_depth = get_max_depth(mcts.root)
        print(f"=== Max Depth of Tree: {max_depth} ===")
        print("========================")
