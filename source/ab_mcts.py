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

    def update_global_posterior(self, reward):
        """Incorporate one new evaluation, once, irrespective of path length."""
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("NIG observations must be finite.")
        kappa_n = self.kappa_post + 1
        mu_n = (self.kappa_post * self.mu_post + reward) / kappa_n
        nu_n = self.nu_post + 1
        tau2_n = (self.nu_post * self.tau2_post
                  + self.kappa_post / kappa_n * (reward - self.mu_post) ** 2) / nu_n

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

        # Fixed local prior; global evidence is fused at selection, not replayed here.
        self.mu_prior = global_hyper.mu_0
        self.kappa_prior = global_hyper.kappa_0
        self.nu_prior = global_hyper.nu_0
        self.tau2_prior = global_hyper.tau2_0

        # Posterior (starts as prior)
        self.mu_post = self.mu_prior
        self.kappa_post = self.kappa_prior
        self.nu_post = self.nu_prior
        self.tau2_post = self.tau2_prior

        self.rewards = []
        self.depth = getattr(parent, "depth", 0) + 1

    def update_posterior(self, new_reward):
        new_reward = float(new_reward)
        self.rewards.append(new_reward)
        N = len(self.rewards)

        r_bar = np.mean(self.rewards)

        mu0, kappa0 = self.mu_prior, self.kappa_prior
        nu0, tau20 = self.nu_prior, self.tau2_prior

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

        # The virtual root has no heuristic to modify; E1 grows its initial pool.
        operators = ['e1'] if is_root else ['counter', 'e2', 'm1', 'm2', 's1']
        for llm in llm_model_names:
            for op in operators:
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
        self._generation_operator = None
        self._backpropagated = False

    def add_child(self, child_node, generation_method=None, generation_action=None):
        child_node.parent = self
        child_node.depth = self.depth + 1
        if generation_method is not None:
            child_node._generation_method = generation_method
        if generation_action is not None:
            child_node._generation_action = generation_action
        self.children.append(child_node)

    def sample_from_node_posterior(self):
        sigma2 = invgamma.rvs(a=self.nu_post / 2.0, scale=self.nu_post * self.tau2_post / 2.0)
        mu_sample = norm.rvs(loc=self.mu_post, scale=np.sqrt(sigma2 / self.kappa_post))
        # theta ~ N(mu, sigma2)
        return norm.rvs(loc=mu_sample, scale=np.sqrt(sigma2))

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
        """
        Score one GEN node; lower is better.
        (mu_g, sigma2_g) ~ NIG_g is combined with the global sample (mu_0, sigma2_0) by precision weighting:
            lambda = (kappa_g / sigma2_g) / (kappa_g / sigma2_g + kappa_0 / sigma2_0)
            mu     = lambda * mu_g + (1 - lambda) * mu_0
            sigma2 = 1 / (kappa_g / sigma2_g + kappa_0 / sigma2_0)
            theta  ~ N(mu, sigma2)
        Without posterior sampling, posterior point estimates replace the samples.
        """
        if ablation.posterior_sampling:
            sigma2_local = invgamma.rvs(a=gen_node.nu_post / 2.0, scale=gen_node.nu_post * gen_node.tau2_post / 2.0)
            mu_local = norm.rvs(loc=gen_node.mu_post, scale=math.sqrt(sigma2_local / gen_node.kappa_post))
        else:
            mu_local, sigma2_local = gen_node.mu_post, gen_node.tau2_post
        local_precision = gen_node.kappa_post / sigma2_local

        if ablation.global_sharing:
            # Precision = kappa / sigma2
            mu_global, sigma2_global = global_params
            global_precision = self.global_hyper.kappa_post / sigma2_global
        else:
            # lambda = 1: the GEN node relies on its local posterior only
            mu_global, global_precision = 0.0, 0.0

        combined_precision = local_precision + global_precision
        lam = local_precision / combined_precision
        combined_mu = lam * mu_local + (1 - lam) * mu_global

        if ablation.posterior_sampling:
            # Sample theta_i ~ Normal(combined_mu, combined_sigma2)
            value = norm.rvs(loc=combined_mu, scale=np.sqrt(1.0 / combined_precision))
        else:
            value = lcb_score(combined_mu, gen_node.visits, self.visits, np.sqrt(self.tau2_post), ablation.ucb_c)

        score = value

        print(
            f"GEN {gen_node.operator_name}: mu_post={gen_node.mu_post:.3f}, tau2={gen_node.tau2_post:.3f}, combined_mu={combined_mu:.3f}, value={value:.3f}, score={score:.3f}")
        return score

    def hierarchical_thompson_sampling(self, fe, num_samples=1, ablation=None,
                                      allow_recombination=True):
        """
        Perform proper Hierarchical Thompson Sampling:
        1. Sample global hyperparameter φ ~ p(φ | D)
        2. For each GEN node G_i, sample θ_i ~ p(θ_i | φ, D_i)
        3. Choose the GEN with the best sampled reward
        The ablation switches replace individual steps (see AblationConfig).
        """
        ablation = ablation or AblationConfig()

        # 1. Sample global hyperparameter (posterior mean when sampling is disabled)
        if ablation.posterior_sampling and ablation.global_sharing:
            global_params = self.global_hyper.sample_hyperparameter()
        else:
            global_params = (self.global_hyper.mu_post, self.global_hyper.tau2_post)

        # 2. Score GEN actions
        gen_candidates = []
        for gen_node in self.gen_nodes:
            # S1 needs at least two real heuristics on the selected path.
            if gen_node.operator_name == 's1' and self.depth < 2:
                continue
            if gen_node.operator_name == 'e2' and not allow_recombination:
                continue
            score = self.gen_score(gen_node, fe, global_params, ablation)
            action_identifier = (gen_node.llm_model_name, gen_node.operator_name)
            gen_candidates.append(('GEN', action_identifier, score))

        if gen_candidates and not ablation.adaptive_op_selection:
            # Expand-vs-continue still uses the best GEN score; only the operator is random
            best_gen_score = min(c[2] for c in gen_candidates)
            chosen = random.choice(gen_candidates)
            gen_candidates = [('GEN', chosen[1], best_gen_score)]

        # 3. Choose the best action (lowest reward)
        candidates = gen_candidates + self.cont_candidates(num_samples, ablation)
        best_candidate = min(candidates, key=lambda x: x[2])

        return best_candidate


    def select_best_action_via_thompson(self, fe, num_samples=1, ablation=None,
                                      allow_recombination=True):
        return self.hierarchical_thompson_sampling(
            fe, num_samples, ablation, allow_recombination=allow_recombination)



class AB_MCTS_A:
    def __init__(self, root_answer, llm_model_names, ablation=None):
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

    def select_expansion_target(self, fe, population=None):
        current = self.root
        while True:
            action, info, _ = current.select_best_action_via_thompson(
                fe=fe, ablation=self.ablation,
                allow_recombination=population is None or any(
                    item['code'] != current.code for item in population))
            if action == 'GEN':
                return current, action, info
            current = current.children[info]

    def backpropagate(self, node: MCTSNode, op_name):
        """Update every real/GEN node on the path, and global evidence once."""
        if node._backpropagated:
            raise ValueError("This evaluation has already been backpropagated.")
        score = float(node.reward)
        if not math.isfinite(score):
            raise ValueError("NIG observations must be finite.")
        node._generation_operator = op_name
        node._backpropagated = True
        if score not in self.rank_list:
            self.rank_list.append(score)
            self.rank_list.sort()
        self.all_rewards_store[(node._generation_action, op_name)].append(score)

        current = node
        while current is not None:
            current.update_node_posterior(score)
            parent = current.parent
            if parent is not None:
                for gen in parent.gen_nodes:
                    if (gen.operator_name == current._generation_operator
                            and gen.llm_model_name == current._generation_action):
                        gen.update_posterior(score)
                        break
            current = parent
        if self.ablation.global_sharing:
            self.global_hyper.update_global_posterior(score)
