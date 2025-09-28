import random
import math
from collections import defaultdict
import numpy as np
from scipy.stats import invgamma

class GENNode:
    def __init__(self, parent, llm_model_name, visits=0):
        self.parent = parent
        self.llm_model_name = llm_model_name
        self.visits = visits
        self.mu_prior = parent.reward if parent.is_root == True else 0
        self.kappa_prior = 1.0
        self.nu_prior = 1.0
        self.tau2_prior = 1.0
        self.mu_post = self.mu_prior
        self.kappa_post = self.kappa_prior
        self.nu_post = self.nu_prior
        self.tau2_post = self.tau2_prior
        self.rewards = []

    def sample_from_posterior(self):

        sigma2 = invgamma.rvs(max(self.nu_post / 2, 0.1), scale=max(self.nu_post * self.tau2_post / 2, 0.1))
        kappa_post = max(self.kappa_post, 1e-6)
        mu = np.random.normal(self.mu_post, math.sqrt(max(sigma2 / kappa_post, 1e-12)))
        return mu


    def update_posterior(self, new_reward):
        self.visits += 1
        self.rewards.append(float(new_reward))
        rewards = self.rewards
        N = len(rewards)
        r_bar = np.mean(rewards) if N > 0 else float(new_reward)

        kappa_prior = self.kappa_prior
        nu_prior = self.nu_prior
        tau2_prior = self.tau2_prior
        mu_prior = self.mu_prior

        self.kappa_post = kappa_prior + N
        if (kappa_prior + N) > 0:
            self.mu_post = (kappa_prior * mu_prior + N * r_bar) / (kappa_prior + N)
        else:
            self.mu_post = mu_prior

        self.nu_post = nu_prior + N

        if N > 1:
            sum_sq = sum((r - r_bar) ** 2 for r in rewards)
        else:
            sum_sq = 0.0

        if (kappa_prior + N) > 0:
            term2 = (N * kappa_prior) / (kappa_prior + N) * (self.mu_post - r_bar) ** 2
        else:
            term2 = 0.0

        if self.nu_post > 0:
            self.tau2_post = (nu_prior * tau2_prior + sum_sq + term2) / self.nu_post
        else:
            self.tau2_post = tau2_prior


class MCTSNode:
    def __init__(self, algorithm, code, obj, depth=0, is_root=False, parent=None, visit=0, raw_info=None,
                 llm_model_names=None):

        self.is_root = is_root
        self.algorithm = algorithm
        self.code = code
        self.parent = parent.mu_prior if is_root == False else 0
        self.depth = depth
        self.children = []
        self.visits = visit
        self.raw_info = raw_info
        self.subtree = []
        self.reward = float(obj)
        self.children_info = []

        # Create GEN nodes for this node
        self.gen_nodes = [GENNode(self, m) for m in llm_model_names] if llm_model_names else []

        # Each node maintains its own posterior for CONT actions (representing refinement)
        self.mu_prior = 7
        self.kappa_prior = 1.0
        self.nu_prior = 3.0
        self.tau2_prior = 1.0
        self.mu_post = self.mu_prior
        self.kappa_post = self.kappa_prior
        self.nu_post = self.nu_prior
        self.tau2_post = self.tau2_prior
        self.node_rewards = []

        # Track generation metadata
        self._generation_method = None
        self._generation_action = None

    def add_child(self, child_node, generation_method=None, generation_action=None):
        """Add a child node and set up generation tracking."""
        child_node.parent = self
        child_node.depth = self.depth + 1
        child_node._generation_method = generation_method
        child_node._generation_action = generation_action
        self.children.append(child_node)

    def sample_from_gen_node(self, gen_node: GENNode):
        return gen_node.sample_from_posterior()

    def sample_from_node_posterior(self):
        """Sample from this node's posterior (used for CONT actions)."""

        sigma2 = invgamma.rvs(max(self.nu_post / 2, 0.1), scale=max(self.nu_post * self.tau2_post / 2, 0.1))
        kappa_post = max(self.kappa_post, 1e-6)
        mu = np.random.normal(self.mu_post, math.sqrt(max(sigma2 / kappa_post, 1e-12)))
        return mu

    def update_node_posterior(self, new_reward):
        """Update this node's posterior distribution."""
        self.visits += 1
        self.node_rewards.append(float(new_reward))
        rewards = self.node_rewards
        N = len(rewards)
        r_bar = np.mean(rewards) if N > 0 else float(new_reward)

        kappa_prior = self.kappa_prior
        nu_prior = self.nu_prior
        tau2_prior = self.tau2_prior
        mu_prior = self.mu_prior

        self.kappa_post = kappa_prior + N
        self.mu_post = (kappa_prior * mu_prior + N * r_bar) / (kappa_prior + N) if (kappa_prior + N) > 0 else mu_prior
        self.nu_post = nu_prior + N

        sum_sq = sum((r - r_bar) ** 2 for r in rewards) if N > 1 else 0.0
        term2 = (N * kappa_prior) / (kappa_prior + N) * (self.mu_post - r_bar) ** 2 if (kappa_prior + N) > 0 else 0.0
        self.tau2_post = (nu_prior * tau2_prior + sum_sq + term2) / self.nu_post if self.nu_post > 0 else tau2_prior

    def select_best_action_via_thompson(self, num_samples=1, epsilon=0.3):

        candidates = []

        if random.random() < epsilon:

            # CONT actions
            for i, child in enumerate(self.children):
                reward = child.reward
                candidates.append(('CONT', i, reward))

            if not candidates:
                if self.gen_nodes:
                    return 'GEN', self.gen_nodes[0].llm_model_name, float('inf')
                return 'GEN', None, float('inf')

            best_candidate = min(candidates, key=lambda x: x[2])
            return best_candidate


        else:

            """Use Thompson sampling to choose between GEN and CONT actions."""

            # Sample from GEN nodes
            for gen_node in self.gen_nodes:
                samples = [self.sample_from_gen_node(gen_node) for _ in range(num_samples)]
                avg_sample = np.mean(samples)
                candidates.append(('GEN', gen_node.llm_model_name, avg_sample))

            # Sample from existing children (CONT actions)
            for i, child in enumerate(self.children):
                samples = [child.sample_from_node_posterior() for _ in range(num_samples)]
                avg_sample = np.mean(samples)
                candidates.append(('CONT', i, avg_sample))

            if not candidates:
                # Fallback: if no candidates, return GEN with first available model
                if self.gen_nodes:
                    return 'GEN', self.gen_nodes[0].llm_model_name, float('inf')
                return 'GEN', None, float('inf')

            # Select the candidate with the best (lowest) sampled value
            best_candidate = min(candidates, key=lambda x: x[2])

            return best_candidate


    def __repr__(self):
        return f"MCTSNode(code={self.code[:20]}..., reward={self.reward:.4f}, visits={self.visits})"


class AB_MCTS_A:
    def __init__(self, root_answer, llm_model_names, max_depth=10):
        self.max_depth = max_depth
        self.rank_list = []
        self.eval_times = 0
        self.llm_model_names = llm_model_names
        self.all_rewards_store = defaultdict(list)
        self.root = MCTSNode(algorithm=root_answer, code="Root", obj=0, depth=0, is_root=True,
                             llm_model_names=llm_model_names)

    def select_expansion_target(self):
        """
        Select target node for expansion using Thompson sampling.
        Returns (node, action_type, action_info) where:
        - node: the node to expand from
        - action_type: 'GEN' or 'CONT'
        - action_info: model_name for GEN, child_index for CONT
        """
        current_node = self.root

        # Special handling for root - use simple selection if children exist
        # if current_node == self.root and current_node.children:
        #     # Select child based on reward-weighted probability
        #     rewards = [child.reward for child in current_node.children]
        #     weights = [1 / (r + 1e-6) for r in rewards]
        #     total = sum(weights)
        #     probs = [w / total for w in weights]
        #     current_node = random.choices(current_node.children, weights=probs, k=1)[0]

        if current_node == self.root and current_node.children:
            candidates = []

            # CONT actions (các child đã có)
            for i, child in enumerate(current_node.children):
                sample = child.sample_from_node_posterior()  # sample 1 lần
                candidates.append(('CONT', i, sample))

            # chọn candidate tốt nhất theo Thompson sampling
            best_candidate = min(candidates, key=lambda x: x[2])

            current_node = current_node.children[best_candidate[1]]

        # Traverse down the tree using Thompson sampling
        while current_node.depth < self.max_depth:
            if not current_node.children:
                candidates = []
                for gen_node in current_node.gen_nodes:
                    sample = current_node.sample_from_gen_node(gen_node)  # chỉ sample 1 lần
                    candidates.append(('GEN', gen_node.llm_model_name, sample))

                best_candidate = min(candidates, key=lambda x: x[2])
                return current_node, 'GEN', best_candidate[1]

            # Use Thompson sampling to decide between GEN and CONT
            selection_result = current_node.select_best_action_via_thompson()

            action_type, action_info, _ = selection_result

            if action_type == 'GEN':
                return current_node, 'GEN', action_info
            elif action_type == 'CONT':
                child_idx = action_info
                if isinstance(child_idx, int) and 0 <= child_idx < len(current_node.children):
                    current_node = current_node.children[child_idx]
                else:
                    if current_node.gen_nodes:
                        return current_node, 'GEN', current_node.gen_nodes[0].llm_model_name
                    return current_node, 'GEN', self.llm_model_names[0] if self.llm_model_names else None

        # Reached max depth, must expand here
        if current_node.gen_nodes:
            return current_node, 'GEN', random.choice(current_node.gen_nodes).llm_model_name
        return current_node, 'GEN', self.llm_model_names[0] if self.llm_model_names else None


    def backpropagate(self, node: MCTSNode):
        """
        Backpropagate score according to AB-MCTS-A specification:
        1. Update the GEN node that generated this node
        2. Propagate through ancestors updating CONT distributions
        """
        if not hasattr(node, 'reward') or node.reward is None:
            return

        score = float(node.reward)

        # Update rank list for monitoring
        if score not in self.rank_list:
            self.rank_list.append(score)
            self.rank_list.sort()

        generation_method = getattr(node, '_generation_method', None)
        generation_action = getattr(node, '_generation_action', None)

        # Key fix: Only backpropagate if this was a GEN action
        if generation_method == 'GEN' and generation_action and node.parent:
            parent = node.parent

            # Update the GEN node that generated this child
            for gen_node in parent.gen_nodes:
                if gen_node.llm_model_name == generation_action:
                    gen_node.update_posterior(score)
                    break

            # Update rewards store
            self.all_rewards_store[generation_action].append(score)

            # Backpropagate through ancestors (updating CONT distributions)
            current = parent
            while current is not None:
                # Update this node's posterior (represents CONT action from its parent's perspective)
                current.update_node_posterior(score)
                current.visits += 1
                current = current.parent

        # CONT actions don't trigger backpropagation in AB-MCTS-A
        # They just traverse to existing nodes



# from __future__ import annotations
# import random
# import copy
# import math
# from collections import deque, defaultdict
# from enum import Enum
# import tqdm
# import numpy as np
# from scipy.stats import invgamma
# import copy
# import json
# import random
# import time
#
#
# import random
# import math
# from collections import defaultdict
# import numpy as np
# from scipy.stats import invgamma
#
# # ---------- GENNode: posterior (normal-inv-chi2 style) ----------
# class GENNode:
#     def __init__(self, parent, llm_model_name, visits=0):
#         self.parent = parent
#         self.llm_model_name = llm_model_name
#         self.visits = visits
#         # priors
#         self.mu_prior = 0.0
#         self.kappa_prior = 1.0
#         self.nu_prior = 2.0
#         self.tau2_prior = 1.0
#         # posts
#         self.mu_post = self.mu_prior
#         self.kappa_post = self.kappa_prior
#         self.nu_post = self.nu_prior
#         self.tau2_post = self.tau2_prior
#         self.rewards = []
#
#     def sample_from_posterior(self):
#         # If no data, sample from prior-ish broad normal
#         if self.nu_post <= 0 or self.tau2_post <= 0:
#             return np.random.normal(0.0, 1e3)
#         sigma2 = invgamma.rvs(max(self.nu_post / 2, 0.1), scale=max(self.nu_post * self.tau2_post / 2, 0.1))
#         kappa_post = max(self.kappa_post, 1e-6)
#         mu = np.random.normal(self.mu_post, math.sqrt(max(sigma2 / kappa_post, 1e-12)))
#         return mu
#
#     def update_posterior(self, new_reward):
#         self.visits += 1
#         self.rewards.append(float(new_reward))
#         rewards = self.rewards
#         N = len(rewards)
#         r_bar = np.mean(rewards) if N > 0 else float(new_reward)
#
#         kappa_prior = self.kappa_prior
#         nu_prior = self.nu_prior
#         tau2_prior = self.tau2_prior
#         mu_prior = self.mu_prior
#
#         self.kappa_post = kappa_prior + N
#         if (kappa_prior + N) > 0:
#             self.mu_post = (kappa_prior * mu_prior + N * r_bar) / (kappa_prior + N)
#         else:
#             self.mu_post = mu_prior
#
#         self.nu_post = nu_prior + N
#
#         if N > 1:
#             sum_sq = sum((r - r_bar) ** 2 for r in rewards)
#         else:
#             sum_sq = 0.0
#
#         if (kappa_prior + N) > 0:
#             term2 = (N * kappa_prior) / (kappa_prior + N) * (self.mu_post - r_bar) ** 2
#         else:
#             term2 = 0.0
#
#         if self.nu_post > 0:
#             self.tau2_post = (nu_prior * tau2_prior + sum_sq + term2) / self.nu_post
#         else:
#             self.tau2_post = tau2_prior
#
# # ---------- CONTNode: posterior aggregating children rewards ----------
# class CONTNode:
#     def __init__(self, parent, visits=0):
#         self.parent = parent
#         self.visits = visits
#         self.children_rewards = []
#         # priors
#         self.mu_prior = 0.0
#         self.kappa_prior = 1.0
#         self.nu_prior = 2.0
#         self.tau2_prior = 1.0
#         # posts
#         self.mu_post = self.mu_prior
#         self.kappa_post = self.kappa_prior
#         self.nu_post = self.nu_prior
#         self.tau2_post = self.tau2_prior
#
#     def sample_from_posterior(self):
#         if len(self.children_rewards) == 0:
#             # wide uncertainty if no data
#             return np.random.normal(0.0, 1e3)
#         if self.nu_post <= 0 or self.tau2_post <= 0:
#             return np.random.normal(0.0, 1e3)
#         sigma2 = invgamma.rvs(max(self.nu_post / 2, 0.1), scale=max(self.nu_post * self.tau2_post / 2, 0.1))
#         kappa_post = max(self.kappa_post, 1e-6)
#         mu = np.random.normal(self.mu_post, math.sqrt(max(sigma2 / kappa_post, 1e-12)))
#         return mu
#
#     def update_posterior(self):
#         if len(self.children_rewards) == 0:
#             return
#         rewards = [float(x) for x in self.children_rewards]
#         N = len(rewards)
#         r_bar = np.mean(rewards) if N > 0 else 0.0
#
#         kappa_prior = self.kappa_prior
#         nu_prior = self.nu_prior
#         tau2_prior = self.tau2_prior
#         mu_prior = self.mu_prior
#
#         self.kappa_post = kappa_prior + N
#         if (kappa_prior + N) > 0:
#             self.mu_post = (kappa_prior * mu_prior + N * r_bar) / (kappa_prior + N)
#         else:
#             self.mu_post = mu_prior
#         self.nu_post = nu_prior + N
#
#         sum_sq = sum((r - r_bar) ** 2 for r in rewards) if N > 1 else 0.0
#         term2 = (N * kappa_prior) / (kappa_prior + N) * (self.mu_post - r_bar) ** 2 if (kappa_prior + N) > 0 else 0.0
#         if self.nu_post > 0:
#             self.tau2_post = (nu_prior * tau2_prior + sum_sq + term2) / self.nu_post
#         else:
#             self.tau2_post = tau2_prior
#
# # ---------- MCTSNode: tree node ----------
# class MCTSNode:
#     def __init__(self, algorithm, code, obj, depth=0, is_root=False, parent=None, visit=0, raw_info=None, llm_model_names=None):
#         self.algorithm = algorithm
#         self.code = code
#         self.parent = parent
#         self.depth = depth
#         self.children = []
#         self.visits = visit
#         self.raw_info = raw_info
#         self.reward = float(obj)  # minimization: smaller better
#
#         # GEN nodes available at this MCTSNode (one per model)
#         self.gen_nodes = [GENNode(self, m) for m in llm_model_names] if llm_model_names else []
#         # single CONT node that aggregates child performance
#         self.cont_node = CONTNode(self)
#
#         # Node's own posterior (for Thompson selection among children)
#         self.mu_prior = 0.0
#         self.kappa_prior = 1.0
#         self.nu_prior = 2.0
#         self.tau2_prior = 1.0
#         self.mu_post = self.mu_prior
#         self.kappa_post = self.kappa_prior
#         self.nu_post = self.nu_prior
#         self.tau2_post = self.tau2_prior
#         self.node_rewards = []
#
#         # Metadata to track generation
#         # _generation_method: 'GEN' or 'CONT' (or None)
#         # _generation_action: model name string if generated by GEN
#         # _generating_gen_node: reference to the GENNode object that generated this node (if any)
#         self._generation_method = None
#         self._generation_action = None
#         self._generating_gen_node = None
#
#     def add_child(self, child_node: MCTSNode, generation_method=None, generation_action=None, generating_gen_node: GENNode = None):
#         """
#         Add a child and optionally set generation metadata on the child.
#         generation_method: 'GEN' or 'CONT'
#         generation_action: model name used (if GEN)
#         generating_gen_node: reference to GENNode instance that created child (if any)
#         """
#         child_node.parent = self
#         child_node.depth = self.depth + 1
#         child_node._generation_method = generation_method
#         child_node._generation_action = generation_action
#         child_node._generating_gen_node = generating_gen_node
#
#         self.children.append(child_node)
#         # update CONT aggregation
#         if self.cont_node:
#             self.cont_node.children_rewards.append(child_node.reward)
#             self.cont_node.update_posterior()
#
#     # Thompson sample from a GEN node or CONT node
#     def sample_from_gen_node(self, gen_node: GENNode):
#         return gen_node.sample_from_posterior()
#
#     def sample_from_cont(self):
#         return self.cont_node.sample_from_posterior() if self.cont_node else np.random.normal(0.0, 1e3)
#
#     def sample_from_node_posterior(self):
#         # sample single draw from the node's own posterior (used when comparing children)
#         if self.nu_post <= 0 or self.tau2_post <= 0:
#             return np.random.normal(0.0, 1e3)
#         sigma2 = invgamma.rvs(max(self.nu_post / 2, 0.1), scale=max(self.nu_post * self.tau2_post / 2, 0.1))
#         kappa_post = max(self.kappa_post, 1e-6)
#         mu = np.random.normal(self.mu_post, math.sqrt(max(sigma2 / kappa_post, 1e-12)))
#         return mu
#
#     def update_node_posterior(self, new_reward):
#         self.visits += 1
#         self.node_rewards.append(float(new_reward))
#         rewards = self.node_rewards
#         N = len(rewards)
#         r_bar = np.mean(rewards) if N > 0 else float(new_reward)
#
#         kappa_prior = self.kappa_prior
#         nu_prior = self.nu_prior
#         tau2_prior = self.tau2_prior
#         mu_prior = self.mu_prior
#
#         self.kappa_post = kappa_prior + N
#         self.mu_post = (kappa_prior * mu_prior + N * r_bar) / (kappa_prior + N) if (kappa_prior + N) > 0 else mu_prior
#         self.nu_post = nu_prior + N
#
#         sum_sq = sum((r - r_bar) ** 2 for r in rewards) if N > 1 else 0.0
#         term2 = (N * kappa_prior) / (kappa_prior + N) * (self.mu_post - r_bar) ** 2 if (kappa_prior + N) > 0 else 0.0
#         self.tau2_post = (nu_prior * tau2_prior + sum_sq + term2) / self.nu_post if self.nu_post > 0 else tau2_prior
#
#     def update_gen_node_posterior(self, model_name, new_reward):
#         for g in self.gen_nodes:
#             if g.llm_model_name == model_name:
#                 g.update_posterior(new_reward)
#                 break
#
#     def select_best_action_per_llm(self):
#         """
#         For each LLM (GEN candidate), do Thompson sampling once for GEN and once for CONT,
#         and decide whether that model prefers GEN or CONT for this node.
#         Return dict: model_name -> ('GEN' or 'CONT', selected_item, sampled_value)
#         Where selected_item is GENNode or self.cont_node accordingly.
#         """
#         llm_actions = {}
#         # sample once from CONT posterior
#         cont_sample = self.sample_from_cont()
#         for gen_node in self.gen_nodes:
#             gen_sample = self.sample_from_gen_node(gen_node)
#             # minimization: lower sample = better
#             if gen_sample < cont_sample:
#                 llm_actions[gen_node.llm_model_name] = ('GEN', gen_node, gen_sample)
#             else:
#                 llm_actions[gen_node.llm_model_name] = ('CONT', self.cont_node, cont_sample)
#         return llm_actions
#
#     def __repr__(self):
#         return f"MCTSNode(code={self.code[:20]}..., reward={self.reward:.4f}, visits={self.visits})"
#
# # ---------- AB-MCTS-A driver ----------
# class AB_MCTS_A:
#     def __init__(self, root_answer, llm_model_names, max_depth=10):
#         self.max_depth = max_depth
#         self.rank_list = []
#         self.eval_times = 0
#         self.llm_model_names = llm_model_names
#         self.all_rewards_store = defaultdict(list)
#         # root node
#         self.root = MCTSNode(algorithm=root_answer, code="Root", obj=0, depth=0, is_root=True, llm_model_names=llm_model_names)
#
#     # --- Selection: recursive with Thompson sampling ---
#     def select_expansion_target(self):
#         def recursive_select(current_node: MCTSNode):
#             # stop if reached max depth -> prefer GEN at leaf
#             if current_node.depth >= self.max_depth:
#                 return current_node, 'GEN', (self.llm_model_names[0] if self.llm_model_names else None)
#
#             # if no children -> must GEN
#             if not current_node.children:
#                 chosen_model = random.choice(current_node.gen_nodes).llm_model_name if current_node.gen_nodes else (self.llm_model_names[0] if self.llm_model_names else None)
#                 return current_node, 'GEN', chosen_model
#
#             # get per-LLM decision via Thompson sampling
#             llm_actions = current_node.select_best_action_per_llm()
#
#             if llm_actions:
#                 # choose the model whose sampled value is best (min for minimization)
#                 best_model_name = min(llm_actions.keys(), key=lambda m: llm_actions[m][2])
#                 action_type, selected_item, sampled_reward = llm_actions[best_model_name]
#
#                 if action_type == 'GEN':
#                     # expand current_node by GEN using best_model_name
#                     return current_node, 'GEN', best_model_name
#                 else:
#                     # CONT chosen: now select which child to continue from using Thompson sampling over children
#                     # For each child, sample from child.node posterior and choose child with smallest sample
#                     child_samples = []
#                     for child in current_node.children:
#                         s = child.sample_from_node_posterior()
#                         child_samples.append(s)
#                     # pick index of smallest sample (minimization)
#                     chosen_idx = int(np.argmin(child_samples))
#                     chosen_child = current_node.children[chosen_idx]
#                     # continue recursion down the chosen child
#                     return recursive_select(chosen_child)
#             else:
#                 # fallback to GEN with any available gen node
#                 if current_node.gen_nodes:
#                     return current_node, 'GEN', current_node.gen_nodes[0].llm_model_name
#                 return current_node, 'GEN', (self.llm_model_names[0] if self.llm_model_names else None)
#
#         return recursive_select(self.root)
#
#     # --- Backpropagation aligned with paper ---
#     def backpropagate(self, node: MCTSNode):
#         # node.reward must exist
#         if not hasattr(node, 'reward'):
#             return
#
#         score = float(node.reward)
#         # maintain rank list for monitoring
#         if score not in self.rank_list:
#             self.rank_list.append(score)
#             self.rank_list.sort()
#
#         # 1) Update node's own posterior
#         node.update_node_posterior(score)
#
#         # 2) If generated by a GEN node, update that GEN node only
#         gen_node = getattr(node, '_generating_gen_node', None)
#         if gen_node is not None:
#             gen_node.update_posterior(score)
#             self.all_rewards_store[gen_node.llm_model_name].append(score)
#
#             # propagate up from the parent of that gen_node (gen_node.parent)
#             current = gen_node.parent
#             while current is not None:
#                 # update CONT node of this ancestor
#                 if hasattr(current, 'cont_node') and current.cont_node:
#                     current.cont_node.children_rewards.append(score)
#                     current.cont_node.update_posterior()
#
#                 # if this ancestor itself was generated by some GEN node, update that specific GEN node
#                 ancestor_gen_node = getattr(current, '_generating_gen_node', None)
#                 if ancestor_gen_node is not None:
#                     ancestor_gen_node.update_posterior(score)
#                     self.all_rewards_store[ancestor_gen_node.llm_model_name].append(score)
#
#                 # update ancestor's own posterior/statistics
#                 current.update_node_posterior(score)
#                 current.visits += 1
#                 if hasattr(current, 'children') and current.children:
#                     best_child_reward = min(child.reward for child in current.children)
#                     current.reward = min(current.reward, best_child_reward)
#
#                 current = current.parent
#
#         # 3) general upward update along parent chain (ensure CONT nodes and node posteriors are updated)
#         current2 = node.parent
#         while current2 is not None:
#             # update CONT node aggregation
#             if hasattr(current2, 'cont_node') and current2.cont_node:
#                 current2.cont_node.children_rewards.append(score)
#                 current2.cont_node.update_posterior()
#             # update node posterior/statistics
#             current2.update_node_posterior(score)
#             current2.visits += 1
#             if hasattr(current2, 'children') and current2.children:
#                 best_child_reward = min(child.reward for child in current2.children)
#                 current2.reward = min(current2.reward, best_child_reward)
#             current2 = current2.parent
#
#         # 4) ensure root stats updated
#         if self.root is not None:
#             self.root.visits += 1
#             if self.root.children:
#                 best_root_child_reward = min(child.reward for child in self.root.children)
#                 self.root.reward = min(self.root.reward, best_root_child_reward)
#
#     def get_best_node(self):
#         # best child under root (minimization)
#         if not self.root.children:
#             return None
#         return min(self.root.children, key=lambda x: x.reward)



