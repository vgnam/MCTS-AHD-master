import random
import math
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy.stats import invgamma, norm
import warnings


@dataclass
class AblationConfig:
    """Switches for the ablation study; all True = full GENESIS."""
    posterior_sampling: bool = True     # False: deterministic UCB on posterior means
    adaptive_op_selection: bool = True  # False: operator drawn uniformly at random
    global_sharing: bool = True         # False: GEN nodes ignore the global hyper-prior
    ucb_c: float = 1.0                  # exploration constant when posterior_sampling=False


def lcb_score(mean, visits, parent_visits, scale, c):
    """Deterministic UCB score for minimization (lower is better)."""
    return mean - c * scale * math.sqrt(math.log(parent_visits + 1) / (visits + 1))

class GlobalHyperPrior:
    def __init__(self, mu_0=0, kappa_0=1.0, nu_0=2.0, tau2_0=1.0):
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
        # self.reward = float(-100/obj) if is_root == False else 0  # Lower obj is better
        self.reward = obj

        self.children_info = []

        # Global hyper prior
        self.global_hyper = global_hyper

        # Create GEN nodes for this node: one for each (LLM, Operator) pair
        self.gen_nodes = []

        for llm in llm_model_names:
            for op in ['counter', 'e2', 'm1', 'm2', 's1']:
                self.gen_nodes.append(GENNode(self, llm, op, global_hyper))

        # Node posterior (for CONT actions)
        self.mu_prior = 0
        self.kappa_prior = 10
        self.nu_prior = 2.0
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

    def cont_candidates(self, num_samples=1, ablation=None):
        """Score every real child (CONT action); lower is better."""
        ablation = ablation or AblationConfig()
        candidates = []
        for i, child in enumerate(self.children):
            if ablation.posterior_sampling:
                score = np.mean([child.sample_from_node_posterior() for _ in range(num_samples)])
            else:
                score = lcb_score(child.mu_post, child.visits, self.visits, np.sqrt(self.tau2_post), ablation.ucb_c)
            print(f"CHILD {i}: mu_post={child.mu_post:.3f}, tau2={child.tau2_post:.3f}, score={score:.3f}")
            candidates.append(('CONT', i, score))
        return candidates

    def gen_score(self, gen_node, fe, global_params, ablation):
        """Score one GEN node; lower is better."""
        local_precision = gen_node.kappa_post / gen_node.tau2_post
        if ablation.global_sharing:
            # Pull local posterior towards global
            # Precision = 1/variance = kappa / sigma2
            mu_global, sigma2_global = global_params
            global_precision = self.global_hyper.kappa_post / sigma2_global
            combined_precision = local_precision + 0.5 * global_precision
            combined_mu = (gen_node.mu_post * local_precision + mu_global * global_precision) / combined_precision
        else:
            combined_precision = local_precision
            combined_mu = gen_node.mu_post

        if ablation.posterior_sampling:
            # Sample theta_i ~ Normal(combined_mu, combined_sigma2)
            value = norm.rvs(loc=combined_mu, scale=np.sqrt(1.0 / combined_precision))
        else:
            value = lcb_score(combined_mu, gen_node.visits, self.visits, np.sqrt(self.tau2_post), ablation.ucb_c)

        # Feature scaling
        score = value * math.exp(gen_node._lambda * fe / 1000)

        print(
            f"GEN {gen_node.operator_name}: mu_post={gen_node.mu_post:.3f}, tau2={gen_node.tau2_post:.3f}, combined_mu={combined_mu:.3f}, value={value:.3f}, score={score:.3f}")
        return score

    def hierarchical_thompson_sampling(self, fe, num_samples=1, ablation=None):
        """
        Perform proper Hierarchical Thompson Sampling:
        1. Sample global hyperparameter φ ~ p(φ | D)
        2. For each GEN node G_i, sample θ_i ~ p(θ_i | φ, D_i)
        3. Choose the GEN with the best sampled reward
        The ablation switches replace individual steps (see AblationConfig).
        """
        ablation = ablation or AblationConfig()

        # 1. Sample global hyperparameter (posterior mean when sampling is disabled)
        if ablation.posterior_sampling:
            global_params = self.global_hyper.sample_hyperparameter()
        else:
            global_params = (self.global_hyper.mu_post, self.global_hyper.tau2_post)

        # 2. Score GEN actions
        gen_candidates = []
        for gen_node in self.gen_nodes:
            score = self.gen_score(gen_node, fe, global_params, ablation)
            action_identifier = (gen_node.llm_model_name, gen_node.operator_name)
            gen_candidates.append(('GEN', action_identifier, score))

        if gen_candidates and not ablation.adaptive_op_selection:
            # Expand-vs-continue still uses the best GEN score; only the operator is random
            best_gen_score = min(c[2] for c in gen_candidates)
            chosen = random.choice(self.gen_nodes)
            gen_candidates = [('GEN', (chosen.llm_model_name, chosen.operator_name), best_gen_score)]

        # 3. Choose the best action (lowest reward)
        candidates = gen_candidates + self.cont_candidates(num_samples, ablation)
        best_candidate = min(candidates, key=lambda x: x[2])

        return best_candidate


    def select_best_action_via_thompson(self, fe, num_samples=1, epsilon=0, ablation=None):

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
        if len(self.children) >= 7:
            candidates = self.cont_candidates(num_samples, ablation)
            best_candidate = min(candidates, key=lambda x: x[2])
            return best_candidate

        # Use HTS
        return self.hierarchical_thompson_sampling(fe, num_samples, ablation)




class AB_MCTS_A:
    def __init__(self, root_answer, llm_model_names, max_depth=10, ablation=None):
        self.max_depth = max_depth
        self.ablation = ablation or AblationConfig()
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
            candidates = current_node.cont_candidates(ablation=self.ablation)
            best_candidate = min(candidates, key=lambda x: x[2])
            current_node = current_node.children[best_candidate[1]]

        while current_node.depth < self.max_depth:
            if not current_node.children:
                selection_result = current_node.select_best_action_via_thompson(fe=fe, ablation=self.ablation)
                action_type, action_info, _ = selection_result
                return current_node, action_type, action_info

            selection_result = current_node.select_best_action_via_thompson(fe=fe, ablation=self.ablation)
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

        print("backpropagate")
        score = float(node.reward)
        if score not in self.rank_list:
            self.rank_list.append(score)
            self.rank_list.sort()

        llm_name = node._generation_action

        parent = node.parent

        # Update the GEN node that generated this child
        gen_node_found = False
        for gen_node in parent.gen_nodes:
            if gen_node.operator_name == op_name:
                gen_node.update_posterior(score, self.global_hyper)
                gen_node_found = True
                print(111111111111111111111111111111111)
                break

        # Update rewards store
        self.all_rewards_store[(llm_name, op_name)].append(score)

        # Update global hyper-posterior based on updated GEN nodes
        # (without global sharing it stays at the initial prior, so GEN nodes are independent)
        if self.ablation.global_sharing:
            self.global_hyper.update_global_posterior(parent.gen_nodes)

        # Backpropagate through ancestors (CONT nodes)
        current = parent
        while current is not None:
            current.update_node_posterior(score)
            current.visits += 1
            current = current.parent
