from __future__ import annotations
import random
import copy
import math
from collections import deque
from enum import Enum
import tqdm
import numpy as np


class MCTSNode:
    def __init__(self, algorithm, code, obj, depth=0, is_root=False, parent=None, visit=0, raw_info=None, Q=0):
        self.algorithm = algorithm
        self.code = code
        self.parent = parent
        self.depth = depth
        self.children = []
        self.children_info = []
        self.visits = visit
        self.subtree = []
        self.raw_info = raw_info
        self.Q = Q
        self.reward = -1 * obj

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"


class MCTS:
    def __init__(self, root_answer):
        self.exploration_constant_0 = 0.1 # Paramter \lambda_0
        self.alpha = 0.5 # Paramter \alpha
        self.max_depth = 20
        self.epsilon = 1e-10
        self.discount_factor = 1 # constant as 1
        self.q_min = 0
        self.q_max = -10000
        self.rank_list = []

        self.root = MCTSNode(algorithm=root_answer, code=root_answer, depth=0, obj=0, is_root=True)

        # Logs
        self.critiques = []
        self.refinements = []
        self.rewards = []
        self.selected_nodes = []

    def backpropagate(self, node: MCTSNode):
        if node.Q not in self.rank_list:
            self.rank_list.append(node.Q)
            self.rank_list.sort()
        self.q_min = min(self.q_min, node.Q)
        self.q_max = max(self.q_max, node.Q)
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = parent.Q * (1 - self.discount_factor) + best_child_Q * self.discount_factor
            parent.visits += 1
            if parent.code != 'Root' and parent.parent.code == 'Root':
                parent.subtree.append(node)
            parent = parent.parent

    def uct(self, node: MCTSNode, eval_remain):
        self.exploration_constant = (self.exploration_constant_0) * eval_remain
        return (node.Q - self.q_min) / (self.q_max - self.q_min) + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / node.visits
        )

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        ) or node.code == 'Root'
