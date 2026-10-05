"""Shared evaluation and wall-clock budget for one GENESIS run."""

import time


class BudgetExhausted(RuntimeError):
    pass


class SearchBudget:
    def __init__(self, max_fe, seconds=None):
        if max_fe <= 0 or (seconds is not None and seconds <= 0):
            raise ValueError("Search budgets must be positive (or seconds=None).")
        self.max_fe = int(max_fe)
        self.seconds = seconds
        self.evaluations = 0
        self.deadline = None

    def start(self):
        self.deadline = None if self.seconds is None else time.monotonic() + self.seconds

    def remaining_seconds(self):
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExhausted("Search wall-clock budget exhausted.")
        return remaining

    def check(self):
        self.remaining_seconds()
        if self.evaluations >= self.max_fe:
            raise BudgetExhausted("Function evaluation budget exhausted.")

    def reserve_evaluation(self):
        self.check()
        self.evaluations += 1
        return self.evaluations
