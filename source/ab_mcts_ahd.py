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

import copy
import random
import numpy as np
from math import log
from scipy.spatial.distance import euclidean
from scipy.sparse.csgraph import minimum_spanning_tree


class AB_MCTS_A_AHD:
    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage
        self.paras = paras  # <-- Added here
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

    # def expand(self, mcts, cur_node, nodes_set, option, model_name=None, use_BoN=False):
    #     """Expand using specific operator and LLM model"""
    #     interface_ec = self.interface_ecs.get(model_name,
    #                                           list(self.interface_ecs.values())[0]) if self.interface_ecs else None
    #
    #     # === Chuẩn bị path_set cho các option ===
    #     def get_path_set():
    #         if option == 's1':
    #             path = []
    #             node = copy.deepcopy(cur_node)
    #             while node.code != "Root":
    #                 path.append(node.raw_info)
    #                 node = copy.deepcopy(node.parent)
    #             path = self.manage.population_management_s1(path, len(path))
    #             if len(path) == 1:
    #                 return None
    #             return path
    #         elif option == 'e1':
    #             return [copy.deepcopy(child.subtree[random.choices(range(len(child.subtree)), k=1)[0]].raw_info)
    #                     for child in mcts.root.children]
    #         return nodes_set
    #
    #     path_set = get_path_set()
    #     if path_set is None:
    #         return nodes_set
    #
    #     # === Generate offsprings ===
    #     all_offsprings = []
    #     num_generations = 32 if use_BoN else 1
    #
    #     for i in range(num_generations):
    #         self.eval_times, offspring = interface_ec.evolve_algorithm(
    #             self.eval_times, path_set, cur_node.raw_info,
    #             cur_node.children_info, option, advice=None
    #         )
    #
    #         if offspring is None:
    #             if not use_BoN:
    #                 print(f"Timeout emerge, no expanding with action {option}.")
    #                 return nodes_set
    #             continue
    #
    #         all_offsprings.append(offspring)
    #
    #         # Print progress
    #         if not use_BoN:
    #             prefix = f"Action: {option}, Father is Root" if option == 'e1' else \
    #                 f"Action: {option}, Father Obj: {cur_node.raw_info['objective']}"
    #             print(f"{prefix}, Now Obj: {offspring['objective']}, Depth: {cur_node.depth + 1}")
    #
    #     if not all_offsprings:
    #         return nodes_set
    #
    #     # === Select best offsprings ===
    #     if use_BoN:
    #         all_offsprings = sorted(all_offsprings, key=lambda x: float(x['objective']))[:5]
    #         print(f"Best-of-N: Selected top {len(all_offsprings)} from {num_generations} candidates")
    #
    #     # === Add nodes to tree ===
    #     for offspring in all_offsprings:
    #         if offspring['objective'] == float('inf'):
    #             continue
    #
    #         if self.add2pop(nodes_set, offspring):
    #             nodes_set = self.manage.population_management(nodes_set, min(len(nodes_set), self.pop_size))
    #
    #             new_node = MCTSNode(
    #                 algorithm=offspring['algorithm'], code=offspring['code'],
    #                 obj=float(offspring['objective']), parent=cur_node,
    #                 depth=cur_node.depth + 1, visit=1, raw_info=offspring,
    #                 llm_model_names=self.llm_model_names, global_hyper=mcts.global_hyper
    #             )
    #             new_node._generation_method = 'GEN_BON' if use_BoN else 'GEN'
    #             new_node._generation_action = model_name
    #
    #             cur_node.add_child(new_node)
    #             cur_node.children_info.append(offspring)
    #             mcts.backpropagate(new_node, op_name=option)
    #
    #     return nodes_set

    def expand(self, mcts, cur_node, nodes_set, option, model_name=None, op_w=1):
        """
        Expand using specific operator and LLM model.
        op_w: number of children to generate.
        Then refine the best one, and add all to tree.
        """
        interface_ec = self.interface_ecs.get(model_name,
                                              list(self.interface_ecs.values())[0]) if self.interface_ecs else None

        # === Chuẩn bị path_set cho các option ===
        def get_path_set():
            if option == 's1':
                path = []
                node = copy.deepcopy(cur_node)
                while node.code != "Root":
                    path.append(node.raw_info)
                    node = copy.deepcopy(node.parent)
                path = self.manage.population_management_s1(path, len(path))
                if len(path) == 1:
                    return None
                return path
            elif option == 'e1':
                return [copy.deepcopy(child.subtree[random.choices(range(len(child.subtree)), k=1)[0]].raw_info)
                        for child in mcts.root.children]
            return nodes_set

        path_set = get_path_set()
        if path_set is None:
            return nodes_set

        # === Generate op_w offsprings using evolve_algorithm (no refinement yet) ===
        all_offsprings = []

        for i in range(op_w):
            self.eval_times, offspring = interface_ec.evolve_algorithm(
                self.eval_times, path_set, cur_node.raw_info,
                cur_node.children_info, option, advice=None
            )

            if offspring is None:
                continue  # Bỏ qua cá thể lỗi

            all_offsprings.append(offspring)

            # Print progress
            prefix = f"Action: {option}, Father is Root" if option == 'e1' else \
                f"Action: {option}, Father Obj: {cur_node.raw_info['objective']}"
            print(f"{prefix}, Candidate Obj: {offspring['objective']}, Depth: {cur_node.depth + 1}")

        if not all_offsprings:
            return nodes_set

        # === Chọn cá thể tốt nhất theo objective ===
        best_offspring = min(all_offsprings, key=lambda x: float(x['objective']))

        # offsprings_to_add = list(all_offsprings)
        # # === Tinh chỉnh cá thể tốt nhất bằng EDCRR ===
        # for offspring in all_offsprings:
        try:
            _, refined_offspring = interface_ec.get_offspring(path_set, "refine", father=best_offspring)
            refined_offspring['objective'] = interface_ec.batch_evaluate([refined_offspring['code']], 0)
        except Exception:
            refined_offspring = best_offspring

        # === Add tất cả offsprings vào tree (gồm cả refined_offspring nếu khác best_offspring) ===
        offsprings_to_add = list(all_offsprings)
        offsprings_to_add.append(refined_offspring)

        # Nếu cá thể đã tinh chỉnh khác với best_offspring, thêm vào danh sách
        if refined_offspring and refined_offspring != best_offspring:
            offsprings_to_add.append(refined_offspring)

        for offspring in offsprings_to_add:
            if offspring['objective'] == float('inf'):
                continue

            if self.add2pop(nodes_set, offspring):
                nodes_set = self.manage.population_management(nodes_set, min(len(nodes_set), self.pop_size))

                new_node = MCTSNode(
                    algorithm=offspring['algorithm'], code=offspring['code'],
                    obj=float(offspring['objective']), parent=cur_node,
                    depth=cur_node.depth + 1, visit=1, raw_info=offspring,
                    llm_model_names=self.llm_model_names, global_hyper=mcts.global_hyper
                )
                new_node._generation_method = 'GEN'  # Đánh dấu là cá thể gốc
                new_node._generation_action = model_name

                cur_node.add_child(new_node)
                cur_node.children_info.append(offspring)
                mcts.backpropagate(new_node, op_name=option)

        return nodes_set

    # def calculate_diversity_from_nodes(self, nodes_set, threshold=0.95):
    #     """
    #     Tính diversity từ nodes_set sử dụng hàm từ population_encoding.
    #
    #     Args:
    #         nodes_set: List of node dictionaries
    #         threshold: Similarity threshold for clustering
    #
    #     Returns:
    #         Dictionary chứa SDI, CDI và các thông tin liên quan
    #     """
    #     # Extract codes từ nodes_set
    #     codes = [node['code'] for node in nodes_set if 'code' in node]
    #
    #     # Tạo embeddings
    #     embeddings = []
    #     for code in codes:
    #         processed_code = remove_comments_and_docstrings(format_python_code(code))
    #         embedding = get_embedding(processed_code, model=model, tokenizer=tokenizer, device=device)
    #         embeddings.append(embedding)
    #
    #     # Stack embeddings
    #     embeddings_2d = np.vstack(embeddings)
    #
    #     # Compute similarity matrix
    #     similarity_matrix = compute_cosine_similarity(embeddings_2d)
    #     np.fill_diagonal(similarity_matrix, 1)
    #
    #     # Cluster nodes
    #     clusters = cluster_nodes(similarity_matrix, threshold)
    #
    #     # Calculate Shannon Diversity Index
    #     sdi = calculate_shannon_diversity(clusters, len(embeddings))
    #
    #     # Calculate Code Diversity Index
    #     cdi = total_diversity(embeddings)
    #
    #     return {
    #         'sdi': sdi,
    #         'cdi': cdi,
    #         'num_nodes': len(embeddings),
    #         'clusters': clusters,
    #         'num_clusters': len(clusters),
    #         'similarity_matrix': similarity_matrix,
    #         'error': None
    #     }

    def run(self):
        print("- Initialization Start -")

        # Create interface_ec for each LLM model
        self.interface_ecs = {}
        for model_name in self.llm_model_names:
            config = self.llm_configs.get(model_name, {})
            advisor_configs = None
            debate_rounds = 0


            advisor_configs = self.config["llm_advisor"]["list"]
            debate_rounds = self.config["llm_advisor"].get("debate_rounds", 1)

            # ===== InterfaceEC =====
            interface_ec = InterfaceEC(
                self.m,
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
                use_numba=self.use_numba,
                advisor_configs=advisor_configs,
                debate_rounds=debate_rounds
            )

            print(advisor_configs)

            self.interface_ecs[model_name] = interface_ec
            print(f"Created interface for model: {model_name}")

        mcts = AB_MCTS_A('Root', self.llm_model_names)  # Pass LLM model names to MCTS

        for model_name, interface_ec in self.interface_ecs.items():
            self.eval_times, brothers, offsprings = interface_ec.get_algorithm(self.eval_times, [], "i1")
            if offsprings is not None:
                brothers = [offsprings]  # Initialize brothers list
                new_node = MCTSNode(
                    offsprings['algorithm'],
                    offsprings['code'],
                    float(offsprings['objective']),
                    parent=mcts.root,
                    depth=1,
                    visit=1,
                    raw_info=offsprings,
                    llm_model_names=self.llm_model_names,
                    global_hyper=mcts.global_hyper
                )

                new_node._generation_method = 'GEN'
                new_node._generation_action = model_name

                mcts.root.add_child(new_node)
                mcts.root.children_info.append(offsprings)
                mcts.backpropagate(new_node, "i1")

        # Collect additional offsprings for initialization
        all_additional_offsprings = []
        model_idx = 0
        for i in range(1, self.init_size):
            # Cycle through LLMs for initialization
            for model_name in self.llm_model_names:
                interface_ec = self.interface_ecs.get(model_name, list(self.interface_ecs.values())[
                    0]) if self.interface_ecs else None

                if interface_ec:
                    self.eval_times, brothers, offsprings = interface_ec.get_algorithm(self.eval_times,
                                                                                       brothers if 'brothers' in locals() else [],
                                                                                       "e1")
                    if offsprings is not None and offsprings['objective'] != float('inf'):
                        # Store additional offsprings with their model name
                        all_additional_offsprings.append((offsprings, model_name))

        # Sort additional offsprings by objective value
        sorted_additional_offsprings = sorted(all_additional_offsprings,
                                              key=lambda x: float(x[0]['objective']),
                                              reverse=False)  # Assuming higher is better

        # Select top 5 best additional offsprings
        top_5_additional_offsprings = all_additional_offsprings

        # Add top 5 additional offsprings to MCTS
        for offsprings, model_name in top_5_additional_offsprings:
            if 'brothers' not in locals():
                brothers = []
            brothers.append(offsprings)
            new_node = MCTSNode(offsprings['algorithm'], offsprings['code'], float(offsprings['objective']),
                                parent=mcts.root, depth=1, visit=1,
                                raw_info=offsprings,
                                llm_model_names=self.llm_model_names,
                                global_hyper=mcts.global_hyper)

            new_node._generation_method = 'GEN'
            new_node._generation_action = model_name

            mcts.root.add_child(new_node)
            mcts.root.children_info.append(offsprings)
            mcts.backpropagate(new_node, op_name="e1")



        # # --- Create MCTS instance ---
        # mcts = AB_MCTS_A('Root', self.llm_model_names)
        #
        # # --- Load pre-prepared nodes from JSON file ---
        # import os
        # import json
        #
        # # Lấy folder chứa script
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        #
        # # Kết hợp với file JSON
        # json_path = os.path.join(script_dir, "seed.json")
        # with open(json_path, 'r', encoding='utf-8') as f:
        #     pre_nodes = json.load(f)  # list of dicts
        #
        # brothers = []
        # model_name = "mistral/codestral-latest"  # fixed model name for all nodes
        #
        # for node_info in pre_nodes:
        #     algo = node_info['algorithm']
        #     code = node_info['code']
        #     obj = node_info['objective']
        #
        #     new_node = MCTSNode(
        #         algo,
        #         code,
        #         float(obj),
        #         parent=mcts.root,
        #         depth=1,
        #         visit=1,
        #         raw_info=node_info,
        #         llm_model_names=self.llm_model_names,
        #         global_hyper=mcts.global_hyper
        #     )
        #
        #     new_node._generation_method = 'GEN'
        #     new_node._generation_action = model_name
        #
        #     mcts.root.add_child(new_node)
        #     mcts.root.children_info.append(node_info)
        #     mcts.backpropagate(new_node, op_name="i1")

            # brothers.append(node_info)

        # --- Optional: manage population size ---
        size_act = min(len(brothers), self.pop_size)
        brothers = self.manage.population_management(brothers, size_act)
        nodes_set = brothers if 'brothers' in locals() else []
        size_act = min(len(nodes_set), self.pop_size)
        nodes_set = self.manage.population_management(nodes_set, size_act)

        print("- Initialization Finished - Evolution Start -")

        while self.eval_times < self.fe_max:
            print(f"Current rewards of MCTS nodes: {[round(float(x), 2) for x in mcts.rank_list[:]]}")

            # AB-MCTS-A Multiple LLM selection and expansion
            target_node, expansion_type, info = mcts.select_expansion_target(fe=self.eval_times)

            selected_model_name = info[0]
            op = info[1]

            print(
                f"Iter: {self.eval_times}/{self.fe_max} Type: {expansion_type}, Selected Model: {selected_model_name}")

            # Apply operators according to weights
            # n_op = len(self.operators)
            # for i in range(n_op):
            #     op = self.operators[i]
            #     op_w = self.operator_weights[i] if i < len(self.operator_weights) else 1
            #
            #     print(f"OP: {op} (weight: {op_w})", end="|")
            #
            #     # Apply this operator op_w times with selected LLM
            #     for j in range(op_w):
            #         nodes_set = self.expand(mcts, target_node, nodes_set, op, selected_model_name)

            n_op = len(self.operators)
            i = self.operators.index(op)
            op_w = self.operator_weights[i]
            print(f"OP: {op} (weight: {op_w})", end="|")

            interface_ec = next(iter(self.interface_ecs.values()))
            #
            # # === Tính diversity và quyết định có dùng advice không ===
            # advice = None
            # use_advice = False
            #
            # # ===== Thresholds =====
            # LOW_DIVERSITY_SDI_THRESHOLD = 0.65  # SDI ratio < 65% max
            # MIN_NODES_FOR_DIVERSITY = 40  # cần đủ node để SDI có ý nghĩa
            #
            # use_advice = False
            #
            # # ----- Chỉ tính diversity nếu đủ node -----
            # if len(nodes_set) >= MIN_NODES_FOR_DIVERSITY:
            #     diversity_info = self.calculate_diversity_from_nodes(
            #         nodes_set,
            #         threshold=0.95
            #     )
            #
            #     if diversity_info['error'] is None:
            #         sdi = diversity_info['sdi']
            #         num_nodes = diversity_info['num_nodes']
            #
            #         # ---- Max SDI ----
            #         max_sdi = log(num_nodes) if num_nodes > 1 else 0.0
            #         sdi_ratio = sdi / max_sdi if max_sdi > 0 else 0.0
            #
            #         # ---- Decision (SDI ONLY) ----
            #         if sdi_ratio < LOW_DIVERSITY_SDI_THRESHOLD:
            #             use_advice = True
            #
            # # === Lấy advice từ debate nếu diversity thấp ===
            # if use_advice:
            #     nodes_set = self.expand(mcts, target_node, nodes_set, op, selected_model_name, use_BoN=True)
            # else:
            #     # Apply this operator op_w times with selected LLM
            for j in range(1):
                nodes_set = self.expand(mcts, target_node, nodes_set, op, selected_model_name, op_w=op_w)


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
                json.dump(nodes_set[0], f, indent=5)

            # self.print_tree_rewards(mcts)

        return nodes_set[0]["code"], filename

    def print_tree_rewards(self, mcts, indent=0):

        def print_node(node, level=0):
            spaces = "  " * level
            depth = getattr(node, "depth", level)
            print(
                f"{spaces}Node: {node.code[:30] if hasattr(node, 'code') else 'Root'} "
                f"| Depth: {depth} "
                f"| Reward: {getattr(node, 'reward', 'N/A'):.2f} "
                f"| Visits: {getattr(node, 'visits', 0)} "
                f"| Mu post: {node.mu_post}"
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

