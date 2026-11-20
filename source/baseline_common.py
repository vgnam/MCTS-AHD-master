"""Shared plumbing for the MoH and HiFo-Prompt baselines.

Both baselines run under the same rules as GENESIS: every evaluator invocation
(valid or not) consumes one FE from a shared ``SearchBudget``, the wall-clock
deadline is forwarded to LLM requests and evaluator processes, and the best
evaluated heuristics are written as ``population_generation_<fe>.json`` /
``best_population_generation_<fe>.json`` in the run directory.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from utils.utils import TOKEN_USAGE, chat_completion
from .search_budget import SearchBudget


def make_budget(paras, config) -> SearchBudget:
    seconds = (config or {}).get("search_timeout", 60)
    return SearchBudget(paras.ec_fe_max, seconds)


class BudgetedEvaluator:
    """Evaluates one heuristic at a time and charges one FE per call."""

    def __init__(self, problem, budget: SearchBudget):
        self.problem = problem
        self.budget = budget

    def evaluate(self, code: str):
        """Return the (minimised) objective, or None if the heuristic is invalid."""
        timeout = self.budget.remaining_seconds()
        self.budget.reserve_evaluation()
        objs = self.problem.batch_evaluate([code], 0, timeout=timeout)
        if isinstance(objs, str) or not objs or not np.isfinite(objs[0]):
            return None
        return float(np.round(objs[0], 5))


class EliteArchive:
    """Best distinct heuristics seen so far, deduplicated by code."""

    def __init__(self, size: int, output_path: str):
        self.size = max(1, int(size))
        self.output_path = Path(output_path)
        self.items: list[dict] = []

    def add(self, algorithm, code, objective, **extra):
        if objective is None:
            return
        item = {"algorithm": algorithm, "code": code,
                "objective": float(objective), **extra}
        for i, old in enumerate(self.items):
            if old["code"] == code:
                if item["objective"] < old["objective"]:
                    self.items[i] = item
                break
        else:
            self.items.append(item)
        self.items.sort(key=lambda x: x["objective"])
        del self.items[self.size:]

    def save(self, eval_times: int) -> str:
        self.output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"generation_{eval_times}.json"
        (self.output_path / f"population_{suffix}").write_text(
            json.dumps(self.items, indent=2, default=str), encoding="utf-8")
        best = self.output_path / f"best_population_{suffix}"
        best.write_text(json.dumps(self.items[0] if self.items else {}, indent=2, default=str),
                        encoding="utf-8")
        TOKEN_USAGE.checkpoint(eval_times, self.output_path)
        return str(best)


class ChatLLM:
    """Small chat wrapper with a system role, temperature and the search deadline."""

    def __init__(self, model: str, budget: SearchBudget | None = None, batch_size: int = 5,
                 tag: str = "default"):
        self.model = model
        self.tag = tag
        self.budget = budget
        self.batch_size = batch_size
        self.n_calls = 0

    def prompt(self, expertise, message, temperature=1.0):
        if self.budget is not None:
            self.budget.check()
        messages = [{"role": "system", "content": str(expertise or "")},
                    {"role": "user", "content": str(message)}]
        started = time.monotonic()
        with TOKEN_USAGE.tag(self.tag):
            choices = chat_completion(1, messages, temperature=float(temperature), model=self.model,
                                      deadline=self.budget.deadline if self.budget else None)
        self.n_calls += 1
        logging.debug("LLM call %s took %.1fs", self.n_calls, time.monotonic() - started)
        return choices[0].message.content

    def prompt_batch(self, expertise, message_batch, temperature=1.0):
        return [self.prompt(expertise, message, temperature) for message in message_batch]
