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
        self.parent = parent
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
        self.mu_prior = parent.mu_prior if is_root == False else 0
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

        current_node = self.root

        if current_node == self.root and current_node.children:
            candidates = []

            # CONT actions (các child đã có)
            for i, child in enumerate(current_node.children):
                sample = child.sample_from_node_posterior()  # sample 1 lần
                candidates.append(('CONT', i, sample))

            best_candidate = min(candidates, key=lambda x: x[2])

            current_node = current_node.children[best_candidate[1]]

        while current_node.depth < self.max_depth:
            if not current_node.children:
                candidates = []
                for gen_node in current_node.gen_nodes:
                    sample = current_node.sample_from_gen_node(gen_node)  # chỉ sample 1 lần
                    candidates.append(('GEN', gen_node.llm_model_name, sample))

                best_candidate = min(candidates, key=lambda x: x[2])
                return current_node, 'GEN', best_candidate[1]

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




