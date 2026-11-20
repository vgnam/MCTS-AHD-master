"""MoH baseline: Meta-optimization of Heuristics (Shi et al., ICLR 2026).

Port of https://github.com/yiding-s/MoH (``moh.py``) onto this repository's
problem interface and search budget. MoH keeps two populations:

* a *heuristic* population for the task, seeded from LLM-proposed directions;
* a *meta-optimizer* population of Python functions
  ``improve_algorithm(population, utility, language_model, function_format, task)``
  written by the LLM. The utility of a meta-optimizer is the objective of the
  best heuristic it produces when run on the task (``meta_utility``); the current
  meta-optimizer improves the meta-optimizer population itself (self-improvement),
  starting from the hand-written seed optimizers of the reference code.

Deviations required for a like-for-like comparison with GENESIS:

* single task (the configured problem size) instead of a size-weighted set of
  sub-tasks; the paper's cross-size generalisation is therefore not exercised;
* every evaluator call of a heuristic consumes one FE of the shared budget and
  the outer loop runs until the FE / wall-clock budget is exhausted (MoH stops
  on ``max_eval_calls`` too; ``baselines.moh.n_iterations`` caps it if set);
* the reference rejects any heuristic containing the word "random", which is a
  rule of its TSP-GLS prompt only, and is not applied here;
* generated optimizers import the parsing helpers from ``source.moh_utils``.

Generated optimizer code is executed in-process with ``exec``, exactly as in
the reference implementation. Its LLM and evaluator calls honour the budget;
pure-Python infinite loops are filtered only by MoH's own LLM loop-count check.
"""

from __future__ import annotations

import copy
import heapq
import json
import logging
import random
from pathlib import Path

import numpy as np

from .baseline_common import BudgetedEvaluator, ChatLLM, EliteArchive, make_budget
from .moh_utils import clean_code, extract_code, extract_idea, find_txt_block, match_number
from .search_budget import BudgetExhausted

INVALID_UTILITY = 1e10
META_TASK = "meta-optimizer"
EXPERTISE = ("You are an expert in the domain of optimization heuristics and combinatorial optimization "
             "problems. Your task is to design heuristics that can effectively solve optimization problems.")


class _StopSearch(BaseException):
    """Budget exhaustion raised inside generated code.

    Derives from BaseException so that ``except Exception`` blocks written by the
    LLM inside a meta-optimizer cannot swallow the end of the search.
    """


# --------------------------------------------------------------------------
# Reference prompts and seed optimizers (prompts/meta, prompts/helper, problems/meta)
# --------------------------------------------------------------------------

META_DESC = (
    "Task: You should design an efficient metaheuristic using the following constraints in a function format. "
    "Your solution code should balance exploration and exploitation creatively. \n"
    "**First, describe your metaheuristic, novel or traditional optimization strategy and main optimization steps in one sentence. "
    "The description must be inside a brace and marked as a comment.** Next, implement it in Python as a function named \"improve_algorithm\". "
    "This function should accept 5 inputs: 'population', 'utility', 'language_model','function_format','task'. "
    "The function should return 3 outputs: 'best_idea','best_solution','best_utility'. 'utility' is a function that evaluates solutions based on a score function, "
    "'function_format' is the format for model responses, 'task' is the name of the problem to be optimized. "
    "The function returns 'best_idea','best_solution','best_utility' which are the idea behind best solution, best code together with its utility.\n"
    "'language_model' is an instance of the language model class used for code generation, with function "
    "\"def prompt_batch(expertise,message_batch,temperature), return responses_list\" for multiple request to LLM and "
    "\"def prompt(self, expertise, message, temperature) return result\" for single request to LLM, "
    "'population' is a dictionary of several historical best solutions of the task, you only use the following functions to operate: "
    "\"def get_solution_by_index(self, task_name, index): return item\", to get a solution by its utility rank; "
    "\"def get_random_solution(self, task_name): return item\", to get a random solution from the population; "
    "the item returned above is a dictionary with keys 'best_sol' and 'utility'. Other functions you can use are: "
    "\"def get_subtask_size(self, task_name):\" to return the size of the population. \n"
    "Please make sure to include all necessary package imports. Specifically, always add the line "
    "\"from source.moh_utils import extract_code,extract_idea\\n import json\\n\" at the beginning of the code to ensure it runs correctly.\n"
    "**You should use your proposed novel or traditional optimization strategy to select solution in the population, generate different or modified directions, then generate better solutions** "
    "You are free to create metaheuristic or totally novel optimization frameworks under this function signature to optimize solutions. "
    "Your should develop self-improvement or traditional optimization metaheuristics that can refine the solution by adapting, exploring different possibilities, and enhancing performance. "
    "You are encouraged to write different prompts utilizing language model to create innovative prompts for proposing optimization strategies and directions for generating diverse enough directions and better code. "
    "Do not restrict yourself to the given parameters; you can create new ones or alternatives. "
    "You must use the language model to generate code snippets or directions for optimization instead of modifying solutions locally. "
    "Avoid modifying snippets of the solution code, as it may invalidate them. Avoid brute-force or exhaustive search methods. "
    "Never generate infinite loops of code that runs indefinitely. Your solution must not rely on repetitive iterations at any given level of the algorithm. "
    "This applies to the outer loop controlling the overall process (e.g., while loop)."
)

CHECK_ITER = (
    "Your task is to calculate the number of iterations the outer loop will perform, focusing on the behavior and "
    "parameters of any for loops, while loops, or similar constructs within the function. Please give your analysis "
    "step by step and put your final result in an integer in a txt codeblock. Here is the code snippet: {code}"
)

SUBTASK_PLAN = (
    "Provide several high-level directions for improving the heuristic solution to the problem and corresponding "
    "utility above, each aimed at minimizing utility as a result. You can choose to propose suggestions to improve the "
    "current solution (Start with label 'Refinement') or propose alternative novel approaches (Start with label "
    "'Innovation'). Format your response as a JSON codeblock below, including the appropriate label for each suggestion.\n"
    "{\n\"direction\": [\n"
    "    {\"content\": \"Refinement: Your first piece of direction suggestion here.\"},\n"
    "    {\"content\": \"Refinement: Your second piece of direction suggestion here.\"},\n"
    "    {\"content\": \"Innovation: Your third piece of direction suggestion here.\"},\n"
    "    {\"content\": \"Innovation: Your fourth piece of direction suggestion here.\"}\n]\n}"
)

SEED_OPTIMIZER = '''from source.moh_utils import extract_code, extract_idea
import json

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = "You are an expert in the domain of designing meta optimization strategy and combinatorial optimization problems. Your task is to design heuristics that can effectively solve optimization problems."

    # Step 1: Select a random solution from the population
    selected_solution = population.get_random_solution(task)

    # Step 2: Generate directions for improvement
    direction_prompt = (
        f"Given the following heuristic for subtask: {selected_solution['best_sol']} with its idea: {selected_solution['idea']} and utility score: {selected_solution['utility']}, "
        "Summarize the key idea from this heuristic, then provide several totally different ideas from the given one to design improved algorithms with lower utility score. "
        "Provide a single string as the answer, less than 50 words. Your response should be formatted as a json structure: "
        "```json\\n{{\\"insights\\":[\\"content\\",\\"content\\",\\"content\\", ... ,\\"content\\"]}}\\n```."
    )
    response = language_model.prompt(expertise, direction_prompt, temperature=1)
    directions = json.loads(extract_code(response))["insights"]

    # Step 3: Create messages based on generated directions
    message_batch = []
    for direction in directions:
        message = (
            f"Improve the following solution:\\n"
            f"```python\\n{selected_solution}\\n```\\n"
            f"You must return an improved solution. Formatted as follows:\\n{function_format}\\n"
            f"To better solve the problem, you are encouraged to develop new solutions based on the direction proposed: {direction}. "
            "You will be evaluated based on a score function. The lower the score, the better the solution.\\n"
            "Your response must firstly provide a summary of the key idea inside a brace and marked as a comment, followed by the code implementation. "
            "Be as creative as you can under the constraints."
        )
        message_batch.append(message)

    # Step 4: Generate new solutions using the language model
    responses = language_model.prompt_batch(expertise, message_batch, temperature=1)
    new_solutions = extract_code(responses)
    new_ideas = extract_idea(responses)

    # Step 5: Evaluate new solutions
    solutions_with_utilities = [(idea, solution, utility(solution, idea, task)) for idea, solution in zip(new_ideas, new_solutions)]
    best_idea, best_solution, best_utility = min(solutions_with_utilities, key=lambda x: x[2])

    return best_idea, best_solution, best_utility
'''

SEED_META_IMPROVER = '''from source.moh_utils import extract_code, extract_idea
import json

def improve_algorithm(population, utility, language_model, function_format, subtask):
    """Improves a solution according to a utility function."""
    expertise = "You are an expert in the domain of designing meta optimization strategy and designing combinatorial optimization problems. Your task is to design heuristics that can effectively solve optimization problems."
    n_messages = language_model.batch_size
    temperature_values = [0.7, 1.0]
    solutions_cache = set()
    new_solutions = []
    new_ideas = []
    utility_cache = {}

    def evaluate_solution(solution, idea):
        if solution not in utility_cache:
            utility_cache[solution] = utility(solution, idea, subtask)
        return utility_cache[solution]

    for temp in temperature_values:
        message_batch = []
        selected_solution = population.get_random_solution(subtask)
        direction_prompt = f"Given the following heuristic for subtask: {selected_solution['best_sol']} with its idea: {selected_solution['idea']}, and utility score: {selected_solution['utility']},\\
        summarize the key idea from this heuristic, then provide several totally different ideas from the given one to design improved algorithms. \\
        Propose some conventional optimization techniques, or some novel metaheuristic strategies that can be used to improve the current solution. For example, Population-Based, Heuristic (Partial) Search Methods, Local Search and Iterative Improvement, Bandit-Based (Exploration-Exploitation) Methods, etc.\\
        Please provide a single string as the answer, less than 50 words. Your response should be formatted as a json structure: ```json\\n{{\\"insights\\":[\\"content\\",\\"content\\",\\"content\\", ... ,\\"content\\"}}\\n```."
        response = language_model.prompt(expertise, direction_prompt)
        directions = json.loads(extract_code(response))["insights"]
        for direction in directions:
            message = f"""Improve the following solution:
        ```python
        {selected_solution}
        ```
        You must return an improved solution. Formatted as follows:
        {function_format}
        To better solve the problem, you are encouraged to develop new solutions base on the direction proposed: {direction},\\
        If you think the direction is a refinement to current solution, just improve the current solution.\\
        If you think the direction is a new idea, you can develop a brand new solution with NO RELATION to the solution given.\\
        You can add appropriate loops in the code for your needs, but not larger than 5.
        You will be evaluated based on a score function. The lower the score, the better the solution.
        Be as creative as you can under the constraints.
        Generate a solution with temperature={temp} that focuses on different aspects of optimization."""
            message_batch.append(message)

        responses = language_model.prompt_batch(expertise, message_batch, temperature=temp)
        generated_solutions = extract_code(responses)
        generated_ideas = extract_idea(responses)
        scored_solutions = [(idea, sol, evaluate_solution(sol, idea)) for sol, idea in zip(generated_solutions, generated_ideas) if sol not in solutions_cache]
        scored_solutions.sort(key=lambda x: x[2], reverse=True)

        top_solutions = scored_solutions[:n_messages]
        for idea, sol, _ in top_solutions:
            new_ideas.append(idea)
            new_solutions.append(sol)
            solutions_cache.add(sol)

    solutions_and_scores = [(idea, solution, evaluate_solution(solution, idea)) for idea, solution in zip(new_ideas, new_solutions)]
    best_idea, best_solution, best_utility = min(solutions_and_scores, key=lambda x: x[2])
    return best_idea, best_solution, best_utility
'''


def load_improver(code: str):
    """Execute optimizer source and return its ``improve_algorithm`` function."""
    namespace = {"__name__": "moh_generated_optimizer"}
    exec(code, namespace)
    function = namespace.get("improve_algorithm")
    if not callable(function):
        raise ValueError("Generated optimizer does not define improve_algorithm.")
    return function


# --------------------------------------------------------------------------
# Population (utils/population.py)
# --------------------------------------------------------------------------

class Pop:
    def __init__(self, name_list, size):
        self.size = size
        self._population = {name: [] for name in name_list}

    def save_solution(self, task_name, idea, solution, utility):
        solution = clean_code(solution)
        entries = self._population[task_name]
        if not isinstance(solution, str) or "def " not in solution or "return " not in solution:
            return False
        if any(entry["best_sol"] == solution for entry in entries):
            return False
        if task_name != META_TASK:  # same utility: keep the shorter code
            for entry in entries:
                if entry["utility"] == utility:
                    if len(solution) < len(entry["best_sol"]):
                        entry["idea"], entry["best_sol"] = idea, solution
                        return True
                    return False
        before = entries[:]
        entries.append({"idea": idea, "best_sol": solution, "utility": utility})
        self._population[task_name] = heapq.nsmallest(min(self.size, len(entries)), entries,
                                                      key=lambda x: x["utility"])
        return before != self._population[task_name]

    def get_solution_by_index(self, task_name, index):
        return self._population[task_name][index]

    def get_best_solution(self, task_name):
        return self._population[task_name][0]

    def get_random_solution(self, task_name):
        entries = self._population[task_name]
        if not entries:
            raise ValueError("The population is empty.")
        ranks = np.arange(1, len(entries) + 1)
        probs = 1 / (ranks + (self.size / 2))
        return entries[np.random.choice(len(entries), p=probs / probs.sum())]

    def get_population(self, task_name):
        return self._population[task_name]

    def get_subtask_size(self, task_name):
        return len(self._population[task_name])

    def to_dict(self):
        return copy.deepcopy(self._population)


class _BudgetedLLM(ChatLLM):
    """LLM client handed to generated optimizers; budget expiry escapes their try/except."""

    def prompt(self, expertise, message, temperature=1.0):
        try:
            return super().prompt(expertise, message, temperature)
        except BudgetExhausted as exc:
            raise _StopSearch(str(exc)) from None


# --------------------------------------------------------------------------
# MoH search loop
# --------------------------------------------------------------------------

class MoH_AHD:
    def __init__(self, paras, problem, select=None, manage=None, **kwargs):
        self.prob = problem
        self.paras = paras
        self.config = kwargs.get('cfg') or {}
        mcfg = dict(((self.config.get('baselines') or {}).get('moh')) or {})

        self.model = paras.llm_model
        self.meta_model = mcfg.get('meta_model') or self.model
        self.pop_size = int(mcfg.get('pop_size', 10))
        self.batch_size = int(mcfg.get('batch_size', 5))
        self.seed_pop_size = int(mcfg.get('seed_pop_size', paras.init_size))
        self.min_base_pop = int(mcfg.get('min_base_pop', 3))
        self.max_outer_iterations = int(mcfg.get('max_outer_iterations', 10))
        self.max_direction_rounds = int(mcfg.get('max_direction_rounds', 20))
        self.max_idle_iterations = int(mcfg.get('max_idle_iterations', 20))
        n_iterations = mcfg.get('n_iterations')
        self.n_iterations = None if n_iterations in (None, -1) else int(n_iterations)

        self.output_path = paras.exp_output_path
        self.budget = make_budget(paras, self.config)
        self.archive = EliteArchive(max(self.pop_size, paras.pop_size), self.output_path)

        prompts = problem.prompts
        self.subtask = f"{problem.problem}-{problem.problem_size}"
        self.task_description = (prompts.get_task() or "").strip()
        n_in, n_out = len(prompts.get_func_inputs()), len(prompts.get_func_outputs())
        inputs = ", ".join(f"'{s}'" for s in prompts.get_func_inputs())
        outputs = ", ".join(f"'{s}'" for s in prompts.get_func_outputs())
        # Equivalent of the reference prompts/<problem>/desc.txt, built from this repo's problem files.
        self.subtask_form = (
            f"Task: {self.task_description}\n"
            "First, describe your new algorithm and main steps in one sentence. The description must be inside a "
            "brace and marked as a comment. Next, implement it in Python as a function named "
            f"{prompts.get_func_name()}. This function should accept {n_in} input(s): {inputs}. The function should "
            f"return {n_out} output(s): {outputs}. {prompts.get_inout_inf().strip()}\n"
            f"Function signature: {prompts.func_signature}\n"
            "Do not change the function name. Remember to use # at the beginning of each comment inside and outside "
            "the codeblock. Put the code in a single ```python code block. Do not give additional explanations."
        )

        self._cur_iter = -1
        seed = self.config.get("seed", 2024)
        random.seed(seed)
        np.random.seed(seed)
        print(f"- MoH parameters loaded: subtask={self.subtask}, pop_size={self.pop_size}, "
              f"batch_size={self.batch_size}, n_iterations={self.n_iterations} -")

    @property
    def eval_times(self):
        return self.budget.evaluations

    # ----- task-level utility ------------------------------------------------

    def evaluate_heuristic(self, algorithm_str, idea=None):
        if not isinstance(algorithm_str, str) or len(algorithm_str) < 30:
            return INVALID_UTILITY
        try:
            objective = self.evaluator.evaluate(algorithm_str)
        except BudgetExhausted as exc:
            raise _StopSearch(str(exc)) from None
        if objective is None:
            return INVALID_UTILITY
        self.archive.add(idea if isinstance(idea, str) else "", algorithm_str, objective,
                         meta_iteration=self._cur_iter)
        return objective

    def _utility(self, algorithm_str, idea=None, problem_type=None):
        return self.evaluate_heuristic(algorithm_str, idea)

    # ----- seed heuristics -----------------------------------------------------

    def _parse_json(self, response):
        block = extract_code(response) or response
        try:
            return json.loads(block)
        except Exception as exc:
            fixed = self.heu_llm.prompt(
                "You are an expert in finding coding errors and debugging.",
                f"There is an exception {exc} while parsing the json file {block}, please check the json file and return a correct one.",
                0.7)
            return json.loads(extract_code(fixed) or fixed)

    def generate_directions(self, previous=None):
        extra = "" if not previous else f"Do not make it the same with previous generated directions: {previous}."
        message = (
            f"The problem is {self.prob.problem} with corresponding size {self.prob.problem_size}. "
            f"According to the task description:{self.task_description} Provide several high-level directions for generating "
            f"the seed prompt, each aimed at minimizing the utility as a result. {extra} Format your response as a JSON codeblock below:\n"
            "{\n\"direction\": [\n    {\"content\": \"Your first piece of direction suggestion here.\"},\n"
            "    {\"content\": \"Your second piece of direction suggestion here.\"},\n"
            "    {\"content\": \"Your third piece of direction suggestion here.\"},\n    ...\n"
            "    {\"content\": \"Your last direction suggestion here.\"}\n]\n}")
        data = self._parse_json(self.heu_llm.prompt(EXPERTISE, message, 0.7))
        return [item["content"] for item in data["direction"]]

    def generate_seed_heuristics(self, target):
        previous = []
        for _ in range(self.max_direction_rounds):
            if self.subtask_pop.get_subtask_size(self.subtask) >= target:
                return
            try:
                directions = self.generate_directions(previous)
            except (_StopSearch, BudgetExhausted):
                raise
            except Exception:
                logging.exception("MoH direction generation failed.")
                continue
            for direction in directions:
                message = (
                    "Write a function that will implement a Python algorithm to solve a problem as well as possible.\n"
                    f"The optimization problem is {self.prob.problem} and the size you should focus on is {self.prob.problem_size}.\n"
                    f"The output function is formatted as follows:\n```python\n{self.subtask_form}\n```\n"
                    f"You should develop the algorithm that follows the direction: {direction}, "
                    "First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. "
                    "Then, provide a Python function that implements the algorithm.")
                try:
                    response = self.heu_llm.prompt(EXPERTISE, message, 0.7)
                    code = clean_code(extract_code(response))
                    idea = extract_idea(response)
                except (_StopSearch, BudgetExhausted):
                    raise
                except Exception:
                    logging.exception("MoH seed heuristic generation failed.")
                    continue
                utility = self.evaluate_heuristic(code, idea)
                if utility < INVALID_UTILITY:
                    self.subtask_pop.save_solution(self.subtask, idea, code, utility)
            previous += directions
        if self.subtask_pop.get_subtask_size(self.subtask) == 0:
            raise RuntimeError("MoH could not produce any valid seed heuristic.")

    # ----- meta level ----------------------------------------------------------

    def get_improver(self, improve_str):
        try:
            improver = load_improver(improve_str)
            return improver(copy.deepcopy(self.subtask_pop), self._utility, self.heu_llm,
                            self.subtask_form, self.subtask)
        except (_StopSearch, BudgetExhausted):
            raise
        except Exception as exc:
            logging.warning("MoH optimizer failed on the task: %r", exc)
            return exc

    def meta_utility(self, improve_str, idea=None, task=None):
        if not isinstance(improve_str, str) or not improve_str:
            return INVALID_UTILITY
        idea = idea if isinstance(idea, str) else ""
        if "language_model.prompt" not in improve_str and "language_model.prompt_batch" not in improve_str:
            logging.info("MoH optimizer rejected: no prompting step.")
            return INVALID_UTILITY
        check = self.meta_llm.prompt(
            "You will act as a professional coder and analyze a given function provided as a string.",
            CHECK_ITER.replace("{code}", improve_str))
        iterations = match_number(find_txt_block(check))
        if iterations > self.max_outer_iterations:
            logging.info("MoH optimizer rejected: %s outer iterations.", iterations)
            return INVALID_UTILITY

        if self.subtask_pop.get_subtask_size(self.subtask) < self.min_base_pop:
            self.generate_seed_heuristics(self.min_base_pop)
        result = self.get_improver(improve_str)
        if isinstance(result, Exception):
            return INVALID_UTILITY
        try:
            new_idea, new_code, new_utility = result
            new_utility = float(new_utility)
        except Exception:
            return INVALID_UTILITY
        if not new_code:
            return INVALID_UTILITY
        if new_utility < INVALID_UTILITY:  # do not seed the task population with invalid code
            self.subtask_pop.save_solution(self.subtask, new_idea, new_code, new_utility)
        self.improver_pop.save_solution(META_TASK, idea, improve_str, new_utility)
        self.meta_log.append({"iteration": self._cur_iter, "evaluations": self.eval_times,
                              "meta_utility": new_utility, "idea": idea})
        logging.info("MoH meta_utility=%s (FE=%s)", new_utility, self.eval_times)
        return new_utility

    def try_improvement(self, improver):
        try:
            new_idea, new_code, new_utility = improver(
                copy.deepcopy(self.improver_pop), self.meta_utility, self.meta_llm, META_DESC, META_TASK)
            if not new_utility:
                raise ValueError("Utility is invalid or zero")
        except (_StopSearch, BudgetExhausted):
            raise
        except Exception as exc:
            logging.warning("MoH meta-improvement failed: %r", exc)
            return False, None
        self.improver_pop.save_solution(META_TASK, new_idea, self.improver_str, self.meta_utility_val)
        if self.meta_utility_val > new_utility:
            self.meta_utility_val = new_utility
            self.improver_str = new_code
        return True, new_code

    def _save_state(self):
        directory = Path(self.output_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "moh_state.json").write_text(json.dumps({
            "meta_utility": self.meta_utility_val,
            "improvers": self.improver_pop.get_population(META_TASK),
            "heuristics": self.subtask_pop.get_population(self.subtask),
            "log": self.meta_log,
        }, indent=2, default=str), encoding="utf-8")
        return self.archive.save(self.eval_times)

    def run(self):
        self.budget.start()
        self.evaluator = BudgetedEvaluator(self.prob, self.budget)
        self.heu_llm = _BudgetedLLM(self.model, self.budget, self.batch_size)
        self.meta_llm = self.heu_llm if self.meta_model == self.model else \
            _BudgetedLLM(self.meta_model, self.budget, self.batch_size)
        self.subtask_pop = Pop([self.subtask], self.pop_size)
        self.improver_pop = Pop([META_TASK], self.pop_size)
        self.improver_str = clean_code(SEED_OPTIMIZER)
        self.meta_utility_val = INVALID_UTILITY
        self.meta_log = []
        try:
            self.generate_seed_heuristics(self.seed_pop_size)
            self._save_state()
            self.meta_utility_val = self.meta_utility(self.improver_str, "seed optimizer")
            self.improver_pop.save_solution(META_TASK, "seed optimizer", self.improver_str, self.meta_utility_val)
            improver = load_improver(SEED_META_IMPROVER)
            self._cur_iter = 0
            idle, last_fe = 0, self.eval_times
            while self.n_iterations is None or self._cur_iter < self.n_iterations:
                self.budget.check()
                # Without a wall-clock limit, an optimizer population that never reaches the
                # evaluator (every candidate rejected) would otherwise loop forever.
                idle = idle + 1 if self.eval_times == last_fe else 0
                last_fe = self.eval_times
                if idle > self.max_idle_iterations:
                    logging.warning("MoH: %s meta iterations without any evaluation; stopping.", idle - 1)
                    break
                logging.info("MoH iteration %s (FE=%s)", self._cur_iter, self.eval_times)
                success, new_code = self.try_improvement(improver)
                candidate = new_code if success and new_code else \
                    self.improver_pop.get_best_solution(META_TASK)["best_sol"]
                try:
                    improver = load_improver(candidate)
                except Exception:
                    logging.warning("MoH: new optimizer does not load; keeping the best one.")
                    improver = load_improver(self.improver_pop.get_best_solution(META_TASK)["best_sol"])
                self._save_state()
                self._cur_iter += 1
        except (_StopSearch, BudgetExhausted) as exc:
            logging.info("MoH stopped after %s evaluations: %s", self.eval_times, exc)
        if not self.archive.items:
            raise RuntimeError("Search budget exhausted before any valid heuristic was evaluated.")
        filename = self._save_state()
        return self.archive.items[0]['code'], filename
