import random
import math
from collections import defaultdict
import numpy as np
from scipy.stats import invgamma, norm
import warnings

class GlobalHyperPrior:
    def __init__(self, mu_0=0.0, kappa_0=1.0, nu_0=2.0, tau2_0=1.0):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.tau2_0 = tau2_0

        # Posterior hyperparameters
        self.mu_post = self.mu_0
        self.kappa_post = self.kappa_0
        self.nu_post = self.nu_0
        self.tau2_post = self.tau2_0

    def update_global_posterior(self, gen_nodes):
        """
        Update the global hyper-posterior based on data from all GEN nodes.
        Uses the means and variances of local posteriors.
        """
        if not gen_nodes:
            return

        # Get local mu_post and tau2_post from GEN nodes
        mu_locals = [g.mu_post for g in gen_nodes]
        tau2_locals = [g.tau2_post for g in gen_nodes]

        # Calculate averages for new prior
        mu_avg = np.mean(mu_locals) if mu_locals else self.mu_0
        tau2_avg = np.mean(tau2_locals) if tau2_locals else self.tau2_0

        # Update hyper-posterior parameters (simplified aggregation)
        N = len(gen_nodes)
        r_bar = mu_avg
        ssq = np.var(mu_locals) * N if len(mu_locals) > 1 else 0

        kappa_n = self.kappa_0 + N
        mu_n = (self.kappa_0 * self.mu_0 + N * r_bar) / kappa_n
        nu_n = self.nu_0 + N
        tau2_n = (self.nu_0 * self.tau2_0 + ssq + (self.kappa_0 * N) / kappa_n * (r_bar - self.mu_0) ** 2) / nu_n

        self.mu_post = mu_n
        self.kappa_post = kappa_n
        self.nu_post = nu_n
        self.tau2_post = tau2_n

    def sample_hyperparameter(self):
        # Sample variance (global)
        sigma2 = invgamma.rvs(a=self.nu_post / 2, scale=self.nu_post * self.tau2_post / 2)
        # Sample mean (global)
        mu = norm.rvs(loc=self.mu_post, scale=math.sqrt(sigma2 / self.kappa_post))
        return mu, sigma2




class GENNode:
    def __init__(self, parent, llm_model_name, operator_name, global_hyper, visits=0):
        self.parent = parent
        self.llm_model_name = llm_model_name
        self.operator_name = operator_name
        self.visits = visits
        self.global_hyper = global_hyper  # Reference to GlobalHyperPrior

        # Prior is taken from global hyper
        self.mu_prior = global_hyper.mu_post
        self.kappa_prior = global_hyper.kappa_post
        self.nu_prior = global_hyper.nu_post
        self.tau2_prior = global_hyper.tau2_post

        # Posterior (starts as prior)
        self.mu_post = self.mu_prior
        self.kappa_post = self.kappa_prior
        self.nu_post = self.nu_prior
        self.tau2_post = self.tau2_prior

        self.rewards = []
        self._lambda = 1.2
        self.depth = getattr(parent, "depth", 0) + 1

    def update_posterior(self, new_reward, global_hyper):
        new_reward = float(new_reward)
        self.rewards.append(new_reward)
        N = len(self.rewards)

        r_bar = np.mean(self.rewards)

        # Get prior from global hyper (updated)
        mu0, kappa0, nu0, tau20 = global_hyper.mu_post, global_hyper.kappa_post, global_hyper.nu_post, global_hyper.tau2_post

        kappa_n = kappa0 + N
        mu_n = (kappa0 * mu0 + N * r_bar) / kappa_n
        nu_n = nu0 + N

        ssq = np.sum((np.array(self.rewards) - r_bar) ** 2)
        term = (kappa0 * N / kappa_n) * (r_bar - mu0) ** 2
        tau2_n = (nu0 * tau20 + ssq + term) / nu_n

        self.kappa_post = kappa_n
        self.mu_post = mu_n
        self.nu_post = nu_n
        self.tau2_post = tau2_n

        self.visits += 1




class MCTSNode:
    def __init__(self, algorithm, code, obj, depth=0, is_root=False, parent=None, visit=0, raw_info=None,
                 llm_model_names=None, global_hyper=None):

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

        # Global hyper prior
        self.global_hyper = global_hyper

        # Create GEN nodes for this node: one for each (LLM, Operator) pair
        self.gen_nodes = []
        if llm_model_names and ['counter', 'e2', 'm1', 'm2', 's1']:
            for llm in llm_model_names:
                for op in ['counter', 'e2', 'm1', 'm2', 's1']:
                    self.gen_nodes.append(GENNode(self, llm, op, global_hyper))

        # Node posterior (for CONT actions)
        self.mu_prior = 0.0
        self.kappa_prior = 10
        self.nu_prior = 3.0
        self.tau2_prior = 1.0
        self.mu_post = self.mu_prior
        self.kappa_post = self.kappa_prior
        self.nu_post = self.nu_prior
        self.tau2_post = self.tau2_prior
        self.node_rewards = []

        self._generation_method = None
        self._generation_action = None  # Can store (llm_name, op_name) tuple later if needed

    def add_child(self, child_node, generation_method=None, generation_action=None):
        child_node.parent = self
        child_node.depth = self.depth + 1
        child_node._generation_method = generation_method
        child_node._generation_action = generation_action
        self.children.append(child_node)

    def sample_from_node_posterior(self):
        sigma2 = invgamma.rvs(a=self.nu_post / 2.0, scale=self.nu_post * self.tau2_post / 2.0)
        mu_sample = norm.rvs(loc=self.mu_post, scale=np.sqrt(sigma2 / self.kappa_post))
        return mu_sample

    def update_node_posterior(self, new_reward):
        new_reward = float(new_reward)
        self.node_rewards.append(new_reward)
        self.visits += 1

        rewards = np.array(self.node_rewards, dtype=np.float64)
        N = len(rewards)
        if N == 0:
            return

        r_bar = rewards.mean()
        sum_sq = ((rewards - r_bar) ** 2).sum()

        mu0 = self.mu_prior
        kappa0 = self.kappa_prior
        nu0 = self.nu_prior
        tau2_0 = self.tau2_prior

        kappa_n = kappa0 + N
        mu_n = (kappa0 * mu0 + N * r_bar) / kappa_n
        nu_n = nu0 + N
        tau2_n = (nu0 * tau2_0 + sum_sq + (kappa0 * N) / kappa_n * (r_bar - mu0) ** 2) / nu_n

        self.kappa_post = kappa_n
        self.mu_post = mu_n
        self.nu_post = nu_n
        self.tau2_post = tau2_n

    def hierarchical_thompson_sampling(self, fe, num_samples=1):
        """
        Perform proper Hierarchical Thompson Sampling:
        1. Sample global hyperparameter φ ~ p(φ | D)
        2. For each GEN node G_i, sample θ_i ~ p(θ_i | φ, D_i)
        3. Choose the GEN with the best sampled reward
        """
        # 1. Sample global hyperparameter
        mu_global, sigma2_global = self.global_hyper.sample_hyperparameter()

        candidates = []
        for gen_node in self.gen_nodes:
            # 2. Sample local posterior using hierarchical approach
            # Pull local posterior towards global
            # Weighted combination of local mu_post and global mu_global
            # Precision = 1/variance = kappa / sigma2
            local_precision = gen_node.kappa_post / gen_node.tau2_post
            global_precision = self.global_hyper.kappa_post / sigma2_global
            combined_precision = local_precision + global_precision
            combined_mu = (gen_node.mu_post * local_precision + mu_global * global_precision) / combined_precision
            combined_sigma2 = 1.0 / combined_precision

            # Sample theta_i ~ Normal(combined_mu, combined_sigma2)
            mu_local = norm.rvs(loc=combined_mu, scale=np.sqrt(combined_sigma2))

            # Feature scaling
            sampled_reward = mu_local * math.exp(gen_node._lambda * fe / 1000)

            action_identifier = (gen_node.llm_model_name, gen_node.operator_name)
            candidates.append(('GEN', action_identifier, sampled_reward))

        # Sample from CONT actions as before
        for i, child in enumerate(self.children):
            samples = [child.sample_from_node_posterior() for _ in range(num_samples)]
            avg_sample = np.mean(samples)
            candidates.append(('CONT', i, avg_sample))

        # Choose the best action (lowest reward)
        best_candidate = min(candidates, key=lambda x: x[2])
        return best_candidate

    def compute_wasserstein_barycenter(self):
        """
        Compute the Wasserstein barycenter of GEN node posteriors.
        Returns a representative (mu, sigma2).
        """
        if not self.gen_nodes:
            return None

        mus = [g.mu_post for g in self.gen_nodes]
        sigma2s = [g.tau2_post for g in self.gen_nodes]

        # Weight by visits (more visited nodes have higher weight)
        weights = [g.visits + 1 for g in self.gen_nodes]
        weights = np.array(weights) / sum(weights)

        # Calculate barycenter for mean and variance (Wasserstein-2)
        mu_b = np.average(mus, weights=weights)
        sigma2_b = np.average(sigma2s, weights=weights)

        return mu_b, sigma2_b

    def select_best_action_via_thompson(self, fe, num_samples=1, epsilon=0):
        if random.random() < epsilon:
            candidates = []
            for i, child in enumerate(self.children):
                reward = child.reward
                candidates.append(('CONT', i, reward))
            if not candidates:
                if self.gen_nodes:
                    # Return first GEN node's (LLM, Operator) pair
                    first_gen = self.gen_nodes[0]
                    return 'GEN', (first_gen.llm_model_name, first_gen.operator_name), float('inf')
                return 'GEN', None, float('inf')
            best_candidate = min(candidates, key=lambda x: x[2])
            return best_candidate

        # If too many children, only consider CONT actions
        if len(self.children) >= 8:
            candidates = []
            for i, child in enumerate(self.children):
                samples = [child.sample_from_node_posterior() for _ in range(num_samples)]
                avg_sample = np.mean(samples)
                candidates.append(('CONT', i, avg_sample))
            best_candidate = min(candidates, key=lambda x: x[2])
            return best_candidate

        # Use HTS
        return self.hierarchical_thompson_sampling(fe, num_samples)




class AB_MCTS_A:
    def __init__(self, root_answer, llm_model_names, max_depth=10):
        self.max_depth = max_depth
        self.rank_list = []
        self.eval_times = 0
        self.llm_model_names = llm_model_names
        self.operators = ['counter', 'e2', 'm1', 'm2', 's1']  # List of 5 operators
        self.all_rewards_store = defaultdict(list)

        # Global hyper prior
        self.global_hyper = GlobalHyperPrior()

        self.root = MCTSNode(
            algorithm=root_answer,
            code="Root",
            obj=0,
            depth=0,
            is_root=True,
            llm_model_names=llm_model_names,
            global_hyper=self.global_hyper
        )

    def select_expansion_target(self, fe):
        current_node = self.root

        if current_node == self.root and current_node.children:
            candidates = []
            for i, child in enumerate(current_node.children):
                sample = child.sample_from_node_posterior()
                candidates.append(('CONT', i, sample))
            best_candidate = min(candidates, key=lambda x: x[2])
            current_node = current_node.children[best_candidate[1]]

        while current_node.depth < self.max_depth:
            if not current_node.children:
                selection_result = current_node.select_best_action_via_thompson(fe=fe)
                action_type, action_info, _ = selection_result
                return current_node, action_type, action_info

            selection_result = current_node.select_best_action_via_thompson(fe=fe)
            action_type, action_info, _ = selection_result

            if action_type == 'GEN':
                return current_node, 'GEN', action_info  # action_info is (llm_name, op_name)
            elif action_type == 'CONT':
                child_idx = action_info
                current_node = current_node.children[child_idx]


        # Reached max depth, must expand here
        if current_node.gen_nodes:
            chosen_gen = random.choice(current_node.gen_nodes)
            return current_node, 'GEN', (chosen_gen.llm_model_name, chosen_gen.operator_name)
        return current_node, 'GEN', (
        self.llm_model_names[0], self.operators[0])

    def backpropagate(self, node: MCTSNode, op_name):

        score = float(node.reward)
        if score not in self.rank_list:
            self.rank_list.append(score)
            self.rank_list.sort()

        llm_name = node._generation_action

        parent = node.parent

        # Update the GEN node that generated this child
        gen_node_found = False
        for gen_node in parent.gen_nodes:
            if gen_node.llm_model_name == llm_name and gen_node.operator_name == op_name:
                gen_node.update_posterior(score, self.global_hyper)
                gen_node_found = True
                break

        # Update rewards store
        self.all_rewards_store[(llm_name, op_name)].append(score)

        # Update global hyper-posterior based on updated GEN nodes
        self.global_hyper.update_global_posterior(parent.gen_nodes)

        # Backpropagate through ancestors (CONT nodes)
        current = parent
        while current is not None:
            current.update_node_posterior(score)
            current.visits += 1
            current = current.parent
