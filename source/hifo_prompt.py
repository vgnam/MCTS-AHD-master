"""HiFo-Prompt baseline (Chen et al., ICLR 2026).

Port of https://github.com/Challenger-XJTU/HiFo-Prompt (``methods/hifo``) onto
this repository's problem interface and search budget:

* Hindsight: an ``InsightPool`` of design principles distilled by the LLM from
  the top 30% of the population, scored by the quality of the offspring whose
  prompts used them, with probation and decayed-effectiveness eviction.
* Foresight: an ``EvolutionaryNavigator`` that tracks stagnation, improvement
  and diversity and switches between exploration / exploitation / balanced
  regimes, each with its own pool of design directives.
* EoH-style generations with operators e1, e2, m1, m2, m3; every operator
  produces ``pop_size`` offspring and the population is truncated greedily.

Deviations required for a like-for-like comparison with GENESIS: offspring are
evaluated sequentially through the shared ``Problem`` evaluator, every evaluator
call (valid or not) consumes one FE, generations continue until the FE or
wall-clock budget is exhausted (instead of a fixed ``ec_n_pop``), and the
-0.8 insight penalty for failed offspring is applied (the reference code
intends it but overwrites the metadata before reading it).
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from collections import deque

import numpy as np

from .baseline_common import BudgetedEvaluator, EliteArchive, make_budget
from .evolution import Evolution
from .search_budget import BudgetExhausted

FAILED_OFFSPRING_EFFECTIVENESS = -0.8


# --------------------------------------------------------------------------
# Hindsight: insight pool
# --------------------------------------------------------------------------

class InsightPool:
    DEFAULT_TIPS = [
        "Design adaptive hybrid meta-heuristics synergistically fusing multiple search paradigms and dynamically tune operator parameters based on search stage or problem features",
        "Employ machine learning or pattern recognition to mine deep problem structures and optimal solution patterns then use learned insights to intelligently bias towards promising search regions or constructive choices",
        "Explore objective function engineering by introducing auxiliary or surrogate objectives or by dynamically adjusting weights to reshape the search landscape aiding escape from local optima or guiding diverse exploration",
        "Construct problem specialized efficient solution representations and co design dedicated core operators to fully leverage representation structure for powerful solution space exploration",
        "Implement intelligent diversification and restart strategies based on solution feature space analysis systematically targeting uncovered feature regions to promote global search coverage and escape deep local optima",
    ]
    PROBATION_USAGE_COUNT = 3
    DECAY_RATE = 0.01
    SMOOTHING = 0.3

    def __init__(self, max_size=30, initial_tips=None):
        self.max_size = max_size
        self.tips = deque(maxlen=max_size)
        self.tip_stats = {}
        self.current_generation = 0
        for tip in (initial_tips or self.DEFAULT_TIPS):
            self.add_tip(tip)

    @staticmethod
    def _similarity(tip1, tip2):
        words1, words2 = set(tip1.lower().split()), set(tip2.lower().split())
        if not words1 or not words2:
            return 0
        return len(words1 & words2) / max(len(words1), len(words2))

    def add_tip(self, tip, tags=None):
        if any(self._similarity(tip, existing) > 0.7 for existing in self.tips):
            return False
        if len(self.tips) >= self.max_size:
            self._evict_tips()
        self.tips.append(tip)
        self.tip_stats[tip] = {'used_count': 0, 'effectiveness': 0.0, 'total_effectiveness': 0.0,
                               'last_used_generation': self.current_generation, 'tags': tags or []}
        return True

    def _eviction_score(self, stats):
        if stats['used_count'] < self.PROBATION_USAGE_COUNT:
            return float('inf')  # probation immunity
        idle = self.current_generation - stats.get('last_used_generation', self.current_generation)
        return stats['effectiveness'] - idle * self.DECAY_RATE

    def _evict_tips(self):
        if len(self.tips) < self.max_size:
            return
        candidates = [(tip, self._eviction_score(self.tip_stats[tip])) for tip in self.tips]
        candidates = [(tip, score) for tip, score in candidates if score != float('inf')]
        if candidates:
            victim = min(candidates, key=lambda x: x[1])[0]
        else:
            victim = min(self.tips, key=lambda t: self.tip_stats[t].get('last_used_generation', 0))
        self.tips.remove(victim)
        del self.tip_stats[victim]

    def update_generation(self, generation):
        self.current_generation = generation

    def get_tips(self, k=3):
        """The reference implementation's default "adaptive" strategy."""
        if not self.tips:
            return []
        if len(self.tips) <= k:
            selected = list(self.tips)
        else:
            scored = []
            for tip in self.tips:
                stats = self.tip_stats[tip]
                usage_penalty = -0.1 * math.log(stats['used_count'] + 1)
                recency_bonus = 0.2 if (self.current_generation - stats['last_used_generation']) <= 2 else 0
                scored.append((tip, stats['effectiveness'] + usage_penalty + recency_bonus))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = [tip for tip, _ in scored[:k]]
        for tip in selected:
            self.tip_stats[tip]['used_count'] += 1
            self.tip_stats[tip]['last_used_generation'] = self.current_generation
        return selected

    def update_tip_stats(self, tip, effectiveness):
        stats = self.tip_stats.get(tip)
        if stats is None:
            return
        stats['total_effectiveness'] += effectiveness
        new = (1 - self.SMOOTHING) * stats['effectiveness'] + self.SMOOTHING * effectiveness
        stats['effectiveness'] = max(-1.0, min(1.0, new))
        stats['last_used_generation'] = self.current_generation


# --------------------------------------------------------------------------
# Foresight: evolutionary navigator
# --------------------------------------------------------------------------

class EvolutionaryNavigator:
    REGIMES = ["exploration", "exploitation", "balanced"]
    DESIGN_DIRECTIVES = {
        "general": [
            "optimizing objective function evaluation criteria",
            "considering long-term impact of current decisions",
            "balancing local optimality with global search strategies",
            "improving algorithm robustness across different problem instances",
            "managing computational complexity and time efficiency",
        ],
        "exploitation": [
            "refining core evaluation and scoring functions",
            "fine-tuning critical algorithm parameters and thresholds",
            "optimizing established successful strategies and patterns",
            "reducing unnecessary computational overhead and redundancy",
            "improving precision of existing heuristics and rules",
        ],
        "exploration": [
            "exploring novel solution construction methodologies",
            "investigating alternative problem decomposition approaches",
            "introducing new randomization or adaptive mechanisms",
            "considering completely different algorithmic paradigms",
            "experimenting with hybrid strategy combinations",
        ],
    }

    def __init__(self, stagnation_limit=3, improvement_streak=2, low_diversity=0.3):
        self.stagnation_count = 0
        self.improvement_count = 0
        self.last_best_fitness = None
        self.last_guidance = None
        self.stagnation_limit = stagnation_limit
        self.improvement_streak = improvement_streak
        self.low_diversity = low_diversity

    def get_guidance(self, best_fitness_history=None, diversity_history=None):
        directives = self.DESIGN_DIRECTIVES
        if not best_fitness_history or len(best_fitness_history) < 2:
            return "balanced", random.choice(directives["general"])

        current_best = best_fitness_history[-1]
        if self.last_best_fitness is not None:
            if self.last_best_fitness - current_best <= 1e-4:
                self.stagnation_count += 1
                self.improvement_count = 0
            else:
                self.improvement_count += 1
                self.stagnation_count = 0
        self.last_best_fitness = current_best

        low_diversity = bool(diversity_history) and diversity_history[-1] < self.low_diversity
        if self.stagnation_count >= self.stagnation_limit or low_diversity:
            regime = "exploration"
            directive = random.choice(directives["exploration"])
        elif self.improvement_count >= self.improvement_streak:
            regime = "exploitation"
            directive = random.choice(directives["exploitation"])
        else:
            regime = random.choices(self.REGIMES, weights=[0.25, 0.25, 0.5], k=1)[0]
            if regime == "balanced":
                pool = directives["general"] + directives["exploitation"] + directives["exploration"]
            else:
                pool = directives[regime]
            directive = random.choice(pool)
        self.last_guidance = (regime, directive)
        return regime, directive


# --------------------------------------------------------------------------
# HiFo prompts (hifo_evolution.py)
# --------------------------------------------------------------------------

_REGIME_HINTS = {
    "i1": {"exploration": "Try to explore a significantly different approach compared to conventional solutions.",
           "exploitation": "Focus on refining and optimizing the most effective patterns in optimization algorithms.",
           "balanced": "Strike a balance between novel ideas and proven effective techniques."},
    "e1": {"exploration": "Try to explore a significantly different approach from both the provided algorithms and conventional solutions.",
           "exploitation": "Focus on combining the strengths from the provided algorithms while addressing their limitations.",
           "balanced": "Create a balanced algorithm that introduces some novel ideas while building on proven patterns from the examples."},
    "e2": {"exploration": "Extend the backbone idea in a novel direction that differs from the provided algorithms.",
           "exploitation": "Refine and optimize the backbone idea to create a more effective implementation.",
           "balanced": "Balance between preserving the effective aspects of the backbone while introducing some novel improvements."},
    "m1": {"exploration": "Introduce significant modifications to create a distinctly different variant.",
           "exploitation": "Focus on refining and optimizing the algorithm while preserving its core structure.",
           "balanced": "Modify key components of the algorithm while preserving its effective aspects."},
    "m2": {"exploration": "Try significantly different parameter values or even a completely different parameterization scheme.",
           "exploitation": "Fine-tune the parameters to optimize performance while keeping the same general structure.",
           "balanced": "Modify some parameters substantially while making minor adjustments to others."},
    "m3": {"exploration": "Consider more radical simplifications that may lead to novel but robust approaches.",
           "exploitation": "Focus on precise, targeted simplifications of components that are likely overfit.",
           "balanced": "Find a balance between simplification and preserving the algorithm's proven strengths."},
}
_DIRECTIVE_LEAD = {
    "i1": "For this task, please pay special attention to",
    "e1": "For this exploration, please pay special attention to",
    "e2": "For this recombination, please pay special attention to",
    "m1": "For this mutation, please pay special attention to",
    "m2": "When adjusting parameters, please pay special attention to",
    "m3": "When simplifying, please pay special attention to",
}


class HiFoEvolution(Evolution):
    """Reuses the problem text/LLM plumbing of ``Evolution`` with HiFo's prompts."""

    def _signature(self, lead="Next, implement"):
        return (lead + " it in Python as a function named " + self.prompt_func_name
                + ". This function should accept " + str(len(self.prompt_func_inputs)) + " input(s): "
                + self.joined_inputs + ". The function should return " + str(len(self.prompt_func_outputs))
                + " output(s): " + self.joined_outputs + ". " + self.prompt_inout_inf + " "
                + self.prompt_other_inf + "\n")

    @staticmethod
    def _parents_text(indivs):
        return "".join("No." + str(i + 1) + " algorithm and the corresponding code are: \n"
                       + (ind['algorithm'] or "") + "\n" + ind['code'] + "\n"
                       for i, ind in enumerate(indivs))

    def hifo_prompt(self, operator, parents, insights, design_directive, regime):
        describe = ("First, describe your new algorithm and main steps in one sentence. "
                    "The description must be inside a brace. ")
        if operator == "i1":
            prompt = self.prompt_task + "\n" + describe + self._signature()
        elif operator == "e1":
            prompt = (self.prompt_task + "\nI have " + str(len(parents))
                      + " existing algorithms with their codes as follows: \n" + self._parents_text(parents)
                      + "Please help me create a new algorithm that has a totally different form from the given ones. \n"
                      + describe + self._signature())
        elif operator == "e2":
            prompt = (self.prompt_task + "\nI have " + str(len(parents))
                      + " existing algorithms with their codes as follows: \n" + self._parents_text(parents)
                      + "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"
                      "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. "
                      "The description must be inside a brace. " + self._signature("Thirdly, implement"))
        elif operator in ("m1", "m2"):
            indiv = parents[0]
            ask = ("Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"
                   if operator == "m1" else
                   "Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n")
            prompt = (self.prompt_task + "\nI have one algorithm with its code as follows. "
                      "Algorithm description: " + (indiv['algorithm'] or "") + "\nCode:\n" + indiv['code'] + "\n"
                      + ask + describe + self._signature())
        elif operator == "m3":
            prompt = ("First, you need to identify the main components in the function below. "
                      "Next, analyze whether any of these components can be overfit to the in-distribution instances. "
                      "Then, based on your analysis, simplify the components to enhance the generalization to potential out-of-distribution instances. "
                      "Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged. \n"
                      + parents[0]['code'] + "\n" + self.prompt_inout_inf + "\n")
        else:
            raise ValueError(f"Unknown HiFo operator: {operator}")

        if insights:
            prompt += "\nConsider these successful design principles I've observed recently:\n" + \
                      "\n".join(f"- {tip}" for tip in insights)
        if design_directive:
            prompt += f"\n{_DIRECTIVE_LEAD[operator]}: {design_directive}"
        prompt += "\n" + _REGIME_HINTS[operator].get(regime, "")
        prompt += "\nDo not give additional explanations."
        return prompt

    def hifo_get_alg(self, prompt_content):
        """Response parsing of the reference ``_get_alg`` (greedy braces, EoH code regex)."""
        algorithm, code = [], []
        for _ in range(4):
            response = self.interface_llm.get_response(prompt_content)
            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if not algorithm:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)
            code = re.findall(r"import.*return", response, re.DOTALL) or \
                re.findall(r"def.*return", response, re.DOTALL)
            if algorithm and code:
                break
        if not code:
            raise ValueError("No code found in HiFo response.")
        code_all = code[0] + " " + ", ".join(self.prompt_func_outputs)
        return code_all, (algorithm[0] if algorithm else "").strip()


# --------------------------------------------------------------------------
# HiFo search loop (hifo.py + hifo_interface_EC.py)
# --------------------------------------------------------------------------

class HiFo_AHD:
    DEFAULT_OPERATORS = ['e1', 'e2', 'm1', 'm2', 'm3']

    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage
        self.paras = paras
        self.config = kwargs.get('cfg') or {}
        hcfg = dict(((self.config.get('baselines') or {}).get('hifo')) or {})

        self.model = paras.llm_model
        self.pop_size = int(hcfg.get('pop_size', 8))
        self.m = int(hcfg.get('m', 2))
        if self.m > self.pop_size or self.m < 2:
            self.m = 2
        self.operators = list(hcfg.get('operators', self.DEFAULT_OPERATORS))
        self.operator_weights = list(hcfg.get('operator_weights', [1] * len(self.operators)))
        if len(self.operator_weights) != len(self.operators):
            self.operator_weights = [1] * len(self.operators)
        self.init_rounds = int(hcfg.get('init_rounds', 2))
        self.n_insights = int(hcfg.get('insights_per_prompt', 3))
        self.insight_pool_size = int(hcfg.get('insight_pool_size', 30))
        self.extract_prob = float(hcfg.get('insight_extract_prob', 0.8))
        self.use_insights = bool(hcfg.get('use_insights', True))
        self.use_navigator = bool(hcfg.get('use_navigator', True))

        self.output_path = paras.exp_output_path
        self.budget = make_budget(paras, self.config)
        self.archive = EliteArchive(max(self.pop_size, paras.pop_size), self.output_path)

        seed = self.config.get("seed", 2024)
        random.seed(seed)
        np.random.seed(seed)
        print(f"- HiFo-Prompt parameters loaded: pop_size={self.pop_size}, m={self.m}, "
              f"operators={self.operators}, insights={self.use_insights}, navigator={self.use_navigator} -")

    @property
    def eval_times(self):
        return self.budget.evaluations

    # ----- population statistics / hindsight ------------------------------

    def update_population_metrics(self, pop, max_history=50):
        fitnesses = [ind['objective'] for ind in pop if ind['objective'] is not None]
        if fitnesses:
            self.best_fitness_history.append(min(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            self.best_fitness_history = self.best_fitness_history[-max_history:]
            self.avg_fitness_history = self.avg_fitness_history[-max_history:]
        if len(pop) >= 2:
            pairs = len(pop) * (len(pop) - 1) / 2
            different = sum(pop[i]['algorithm'] != pop[j]['algorithm']
                            for i in range(len(pop)) for j in range(i + 1, len(pop)))
            self.diversity_history.append(different / pairs)
            self.diversity_history = self.diversity_history[-max_history:]

    @staticmethod
    def insight_effectiveness(offspring, population):
        if offspring.get('objective') is None:
            return -0.5
        fitnesses = [ind['objective'] for ind in population if ind['objective'] is not None]
        if not fitnesses:
            return 0.0
        value = offspring['objective']
        best, worst, avg = min(fitnesses), max(fitnesses), sum(fitnesses) / len(fitnesses)
        if worst == best:
            return 0.1
        normalized = (worst - value) / (worst - best)
        if value <= best:
            eff = 0.8 + 0.2 * normalized
        elif value <= avg:
            eff = 0.2 + 0.6 * normalized
        else:
            eff = -0.3 + 0.5 * normalized
        return max(-1.0, min(1.0, eff))

    def extract_insights(self, pop):
        if len(pop) < 3:
            return
        top = sorted(pop, key=lambda x: x['objective'])[:max(1, int(len(pop) * 0.3))]
        prompt = "The following are core descriptions of high-performance optimization algorithms evolved recently:\n"
        for i, ind in enumerate(top):
            description = (ind.get('algorithm') or '').strip()
            if len(description) > 8:
                content = description
            else:
                content = ind.get('code', '')
                if len(content) > 1000:
                    content = content[:800] + "...\n# (truncated for brevity)"
            prompt += f"{i + 1}. Algorithm: {content}\n"
        prompt += ("\nPlease extract 1-2 concise, generic, and performance-positive [design principles] or [effective patterns] from the above algorithms."
                   "\nThese principles should be applicable to various combinatorial optimization problems, not just the specific problem domain."
                   "\nWhen formulating these principles, it is essential to draw insights from *both* the conceptual natural language descriptions *and* their corresponding code implementations. Focus on identifying the underlying strategic design choices and algorithmic methodologies rather than superficial characteristics or specific implementation minutiae."
                   "\nEach principle/pattern should be expressed as an independent sentence in the following format:"
                   "\n- Balance local optimization with global solution structure when making decisions."
                   "\n- Prioritize choices that maintain flexibility for future decision-making steps."
                   "\n- Implement adaptive mechanisms that respond to problem instance characteristics.")
        try:
            response = self.evol.interface_llm.get_response(prompt)
        except BudgetExhausted:
            raise
        except Exception:
            logging.exception("HiFo insight extraction failed.")
            return
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                item = line[2:].strip()
                if len(item) > 10:
                    self.insight_pool.add_tip(item, tags=["extracted", "high_performance"])

    # ----- offspring generation -------------------------------------------

    def _guidance(self):
        if self.use_navigator:
            regime, directive = self.navigator.get_guidance(
                self.best_fitness_history, self.diversity_history)
        else:
            regime, directive = "default", None
        insights = self.insight_pool.get_tips(k=self.n_insights) if self.use_insights else []
        return regime, directive, insights

    def _select_parents(self, pop, operator):
        if operator == "i1":
            return []
        if operator in ("e1", "e2"):
            return self.select.parent_selection(pop, self.m)
        return self.select.parent_selection(pop, 1)

    def _generate(self, pop, operator):
        regime, directive, insights = self._guidance()
        parents = self._select_parents(pop, operator)
        prompt = self.evol.hifo_prompt(operator, parents, insights, directive, regime)
        code, algorithm = self.evol.hifo_get_alg(prompt)
        meta = {'operator': operator, 'insights': insights, 'design_directive': directive,
                'regime': regime, 'timestamp': time.time()}
        return code, algorithm, meta

    def get_offspring(self, pop, operator):
        meta = {'insights': []}
        try:
            code, algorithm, meta = self._generate(pop, operator)
            if any(ind['code'] == code for ind in pop):  # one retry, as in the reference
                code, algorithm, meta = self._generate(pop, operator)
        except BudgetExhausted:
            raise
        except Exception:
            logging.exception("HiFo offspring generation failed (%s).", operator)
            for tip in meta['insights']:
                self.insight_pool.update_tip_stats(tip, FAILED_OFFSPRING_EFFECTIVENESS)
            return None

        objective = self.evaluator.evaluate(code)
        offspring = {'algorithm': algorithm, 'code': code, 'objective': objective,
                     'other_inf': None, 'metadata': meta}
        if objective is None:
            for tip in meta['insights']:
                self.insight_pool.update_tip_stats(tip, FAILED_OFFSPRING_EFFECTIVENESS)
            return None
        effectiveness = self.insight_effectiveness(offspring, pop)
        for tip in meta['insights']:
            self.insight_pool.update_tip_stats(tip, effectiveness)
        self.archive.add(algorithm, code, objective, operator=operator, regime=meta['regime'])
        return offspring

    def get_algorithm(self, pop, operator, out):
        """Append up to ``pop_size`` valid offspring to ``out`` (kept on budget expiry)."""
        self.insight_pool.update_generation(len(self.best_fitness_history))
        for _ in range(self.pop_size):
            self.budget.check()
            offspring = self.get_offspring(pop, operator)
            if offspring is not None:
                out.append(offspring)
        if pop:
            self.update_population_metrics(pop)
            if self.use_insights and np.random.random() < self.extract_prob:
                self.extract_insights(pop)

    def _log_generation(self, generation, population):
        self.prompt_log.append({
            "generation": generation,
            "evaluations": self.eval_times,
            "best_fitness": population[0]["objective"] if population else None,
            "diversity": self.diversity_history[-1] if self.diversity_history else None,
            "insight_count": len(self.insight_pool.tips),
            "recent_insights": list(self.insight_pool.tips)[-3:],
            "navigator_guidance": self.navigator.last_guidance,
        })

    def run(self):
        self.budget.start()
        self.evaluator = BudgetedEvaluator(self.prob, self.budget)
        self.evol = HiFoEvolution('', 'Not used', self.model, False, self.prob.prompts,
                                  use_local_llm=False, url='', budget=self.budget)
        self.insight_pool = InsightPool(max_size=self.insight_pool_size)
        self.navigator = EvolutionaryNavigator()
        self.best_fitness_history, self.avg_fitness_history, self.diversity_history = [], [], []
        self.prompt_log = []

        population = []
        generation = 0
        try:
            rounds = 0
            while rounds < self.init_rounds or not population:
                if rounds >= self.init_rounds + 20:
                    raise RuntimeError("HiFo-Prompt could not produce any valid initial heuristic.")
                batch = []
                try:
                    self.get_algorithm([], 'i1', batch)
                finally:
                    population.extend(batch)
                rounds += 1
            population = self.manage.population_management(population, self.pop_size)
            self.archive.save(self.eval_times)
            while True:
                generation += 1
                for operator, weight in zip(self.operators, self.operator_weights):
                    self.budget.check()
                    if np.random.rand() >= weight:
                        continue
                    batch = []
                    try:
                        self.get_algorithm(population, operator, batch)
                    finally:
                        population.extend(batch)
                        population = self.manage.population_management(
                            population, min(len(population), self.pop_size))
                self._log_generation(generation, population)
                self.archive.save(self.eval_times)
                logging.info("HiFo generation %s finished: best=%s, FE=%s", generation,
                             population[0]['objective'] if population else None, self.eval_times)
        except BudgetExhausted as exc:
            logging.info("HiFo-Prompt stopped after %s evaluations: %s", self.eval_times, exc)
        if not self.archive.items:
            raise RuntimeError("Search budget exhausted before any valid heuristic was evaluated.")
        filename = self.archive.save(self.eval_times)
        (self.archive.output_path / "hifo_prompt_log.json").write_text(
            json.dumps(self.prompt_log, indent=2, default=str), encoding="utf-8")
        return self.archive.items[0]['code'], filename
