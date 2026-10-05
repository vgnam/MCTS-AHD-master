"""Offline regressions for GENESIS search, with no API or benchmark calls."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from pathlib import Path
from source.ab_mcts import AB_MCTS_A, AblationConfig, GlobalHyperPrior, MCTSNode
from source.ab_mcts_ahd import AB_MCTS_A_AHD
from source.evolution import Evolution
from source.evolution_interface import InterfaceEC
from source.getParas import Paras
from source.search_budget import BudgetExhausted, SearchBudget
from source import prob_rank, pop_greedy


@pytest.fixture(autouse=True)
def no_external_api(monkeypatch):
    monkeypatch.setattr('utils.utils.completion',
                        Mock(side_effect=AssertionError('Tests must not call an external API.')))


def add_node(tree, parent, value, operator='m1', model='test'):
    node = MCTSNode('idea', f'code {value}', value, parent=parent,
                    llm_model_names=tree.llm_model_names, global_hyper=tree.global_hyper)
    parent.add_child(node, generation_method='GEN', generation_action=model)
    tree.backpropagate(node, operator)
    return node


def posterior(prior, observations):
    mu, kappa, nu, tau = prior
    values = np.asarray(observations)
    n = len(values)
    mean = values.mean()
    return ( (kappa * mu + values.sum()) / (kappa + n), kappa + n, nu + n,
             (nu * tau + ((values - mean) ** 2).sum()
              + kappa * n / (kappa + n) * (mean - mu) ** 2) / (nu + n))


def params(node):
    return node.mu_post, node.kappa_post, node.nu_post, node.tau2_post


def test_observations_reach_leaf_ancestors_and_correct_model_operator():
    tree = AB_MCTS_A('Root', ['other', 'test'])
    first = add_node(tree, tree.root, 10, 'e1')
    second = add_node(tree, first, 20)
    third = add_node(tree, second, 30, 'm2')
    assert third.node_rewards == [30]
    assert first.node_rewards == [10, 20, 30]
    assert first.visits == 3
    assert tree.root.visits == 3
    assert params(first) == pytest.approx(posterior((0, 10, 2, 1), [10, 20, 30]))
    gen = next(g for g in first.gen_nodes if g.operator_name == 'm1' and g.llm_model_name == 'test')
    other = next(g for g in first.gen_nodes if g.operator_name == 'm1' and g.llm_model_name == 'other')
    assert gen.rewards == [20, 30]
    assert other.rewards == []
    assert params(gen) == pytest.approx(posterior((0, 1, 2, 1), [20, 30]))
    assert params(tree.global_hyper) == pytest.approx(posterior((0, 1, 2, 1), [10, 20, 30]))
    assert third._generation_action == 'test'
    with pytest.raises(ValueError, match='already'):
        tree.backpropagate(third, 'm2')


def test_global_updates_are_order_independent_and_cross_branch():
    first, second = GlobalHyperPrior(), GlobalHyperPrior()
    for value in [10, -5, 30, 4]:
        first.update_global_posterior(value)
    for value in [4, 30, -5, 10]:
        second.update_global_posterior(value)
    assert params(first) == pytest.approx(params(second))
    tree = AB_MCTS_A('Root', ['test'])
    left = add_node(tree, tree.root, 10, 'e1')
    right = add_node(tree, tree.root, 20, 'e1')
    add_node(tree, left, 5)
    add_node(tree, right, 15)
    assert params(tree.global_hyper) == pytest.approx(posterior((0, 1, 2, 1), [10, 20, 5, 15]))


def test_no_global_sharing_keeps_global_prior_but_updates_local_nodes():
    tree = AB_MCTS_A('Root', ['test'], ablation=AblationConfig(global_sharing=False))
    node = add_node(tree, tree.root, 10, 'e1')
    add_node(tree, node, 20)
    assert params(tree.global_hyper) == (0, 1, 2, 1)
    assert node.node_rewards == [10, 20]
    assert next(g for g in node.gen_nodes if g.operator_name == 'm1').rewards == [20]


def test_hts_fuses_sampled_means_and_variances_by_precision(monkeypatch):
    tree = AB_MCTS_A('Root', ['test'])
    gen = tree.root.gen_nodes[0]
    gen.kappa_post = 4
    tree.global_hyper.kappa_post = 3
    monkeypatch.setattr('source.ab_mcts.invgamma.rvs', lambda **kw: 2.)
    normal = Mock(side_effect=[8., 99.])
    monkeypatch.setattr('source.ab_mcts.norm.rvs', normal)
    result = tree.root.gen_score(gen, 0, (2., 6.), AblationConfig())
    assert result == 99
    # Local precision=4/2=2, global precision=3/6=.5.
    assert normal.call_args.kwargs['loc'] == pytest.approx(6.8)
    assert normal.call_args.kwargs['scale'] == pytest.approx(np.sqrt(1 / 2.5))


def test_root_can_expand_and_branching_has_no_seven_child_cap(monkeypatch):
    tree = AB_MCTS_A('Root', ['test'])
    for i in range(8):
        add_node(tree, tree.root, i + 1, 'e1')
    monkeypatch.setattr(MCTSNode, 'gen_score', lambda *args: -100)
    monkeypatch.setattr(MCTSNode, 'sample_from_node_posterior', lambda self: 100)
    target, action, info = tree.select_expansion_target(8)
    assert target is tree.root
    assert (action, info) == ('GEN', ('test', 'e1'))


def test_deep_selection_still_uses_hts(monkeypatch):
    tree = AB_MCTS_A('Root', ['test'])
    current = tree.root
    for i in range(12):
        current = add_node(tree, current, i + 1, 'e1' if current.is_root else 'm1')
    visited = []
    def choose(node, **kwargs):
        visited.append(node.depth)
        return ('CONT', 0, 0) if node.children else ('GEN', ('test', 'm2'), 0)
    monkeypatch.setattr(MCTSNode, 'select_best_action_via_thompson', choose)
    target, _, info = tree.select_expansion_target(12)
    assert target is current
    assert visited == list(range(13))
    assert info == ('test', 'm2')


def test_sampling_ablation_is_deterministic(monkeypatch):
    tree = AB_MCTS_A('Root', ['test'], ablation=AblationConfig(posterior_sampling=False))
    add_node(tree, tree.root, 10, 'e1')
    monkeypatch.setattr('source.ab_mcts.invgamma.rvs', Mock(side_effect=AssertionError('random sample')))
    monkeypatch.setattr('source.ab_mcts.norm.rvs', Mock(side_effect=AssertionError('random sample')))
    assert tree.select_expansion_target(1) == tree.select_expansion_target(1)


def test_random_operator_ablation_excludes_inapplicable_synthesis(monkeypatch):
    tree = AB_MCTS_A('Root', ['test'])
    node = add_node(tree, tree.root, 10, 'e1')
    seen = []
    def choose(candidates):
        seen.extend(c[1][1] for c in candidates)
        return candidates[-1]
    monkeypatch.setattr('source.ab_mcts.random.choice', choose)
    node.hierarchical_thompson_sampling(1, ablation=AblationConfig(adaptive_op_selection=False))
    assert 's1' not in seen
    assert set(seen) == {'counter', 'e2', 'm1', 'm2'}


def make_search(tmp_path, max_fe=10, agre=True):
    paras = Paras()
    paras.set_paras(init_size=5, ec_fe_max=max_fe, llm_model_names=['test'],
                    exp_output_path=str(tmp_path), pop_size=30)
    search = AB_MCTS_A_AHD(paras, None, prob_rank, pop_greedy,
                          cfg={'search_timeout': None, 'ablation': {'agre': agre}})
    search.budget.start()
    return search


def mock_interface(budget, objectives):
    interface = InterfaceEC.__new__(InterfaceEC)
    interface.budget = budget
    interface.interface_eval = SimpleNamespace(batch_evaluate=Mock(side_effect=objectives))
    counter = iter(range(100))
    def generate(pop, operator, **kwargs):
        i = next(counter)
        return None, dict(algorithm='same description', code=f'{operator}_{i}', objective=None)
    interface.get_offspring = Mock(side_effect=generate)
    return interface


def test_expansion_evaluates_four_candidates_and_refines_best_once(tmp_path):
    search = make_search(tmp_path)
    interface = mock_interface(search.budget, [[8], [5], [7], [6], [3]])
    search.interface_ecs = {'test': interface}
    tree = AB_MCTS_A('Root', ['test'])
    population = []
    search.expand(tree, tree.root, population, 'e1', 'test')
    assert search.eval_times == 5
    assert interface.interface_eval.batch_evaluate.call_count == 5
    assert len(tree.root.children) == 5  # Same descriptions do not drop distinct code.
    assert population[0]['objective'] == 3
    call = interface.get_offspring.call_args_list[-1]
    assert call.args[1] == 'refine'
    assert call.kwargs['father']['objective'] == 5
    assert sum(n._generation_method == 'AGRE' for n in tree.root.children) == 1


@pytest.mark.parametrize('result', ['timeout', [float('inf')], [float('nan')]])
def test_invalid_refinement_is_counted_without_reinserting_original(tmp_path, result):
    search = make_search(tmp_path)
    interface = mock_interface(search.budget, [[8], [5], [7], [6], result])
    search.interface_ecs = {'test': interface}
    tree = AB_MCTS_A('Root', ['test'])
    population = []
    search.expand(tree, tree.root, population, 'e1', 'test')
    assert search.eval_times == 5
    assert len(tree.root.children) == 4
    assert population[0]['objective'] == 5


def test_agre_disabled_spends_only_four_evaluations(tmp_path):
    search = make_search(tmp_path, agre=False)
    interface = mock_interface(search.budget, [[8], [5], [7], [6]])
    search.interface_ecs = {'test': interface}
    tree = AB_MCTS_A('Root', ['test'])
    search.expand(tree, tree.root, [], 'e1', 'test')
    assert search.eval_times == 4
    assert all(call.args[1] != 'refine' for call in interface.get_offspring.call_args_list)


def test_budget_exhaustion_preserves_partial_batch_and_blocks_refine(tmp_path):
    search = make_search(tmp_path, max_fe=2)
    interface = mock_interface(search.budget, [[8], [5]])
    search.interface_ecs = {'test': interface}
    tree = AB_MCTS_A('Root', ['test'])
    population = []
    with pytest.raises(BudgetExhausted):
        search.expand(tree, tree.root, population, 'e1', 'test')
    assert search.eval_times == 2
    assert len(tree.root.children) == 2
    assert population[0]['objective'] == 5


def test_failed_evaluation_retries_cannot_exceed_budget():
    budget = SearchBudget(2)
    interface = mock_interface(budget, [[float('inf')], 'timeout'])
    with pytest.raises(BudgetExhausted):
        interface.evolve_algorithm(0, [], None, [], 'm1')
    assert budget.evaluations == 2
    assert interface.interface_eval.batch_evaluate.call_count == 2


def test_counter_uses_selected_parent():
    interface = InterfaceEC.__new__(InterfaceEC)
    interface.evol = SimpleNamespace(counter=Mock(return_value=['code', 'idea']),
                                     post_thought=Mock(return_value='description'))
    father = {'code': 'selected'}
    parents, _ = interface._get_alg([{'code': 'elite'}], 'counter', father=father)
    assert parents == [father]
    interface.evol.counter.assert_called_once_with(father, advice=None)


def test_wall_clock_prevents_new_calls_and_is_forwarded_to_evaluator(monkeypatch):
    now = [100.0]
    monkeypatch.setattr('source.search_budget.time.monotonic', lambda: now[0])
    budget = SearchBudget(10, 60)
    budget.start()
    interface = mock_interface(budget, [[3]])
    now[0] = 120
    interface.evaluate_offspring(0, {'code': 'code'})
    assert interface.interface_eval.batch_evaluate.call_args.kwargs['timeout'] == 40
    now[0] = 160
    with pytest.raises(BudgetExhausted):
        interface.evaluate_offspring(1, {'code': 'code'})
    assert budget.evaluations == 1


def test_complete_run_stops_at_budget_and_saves_best(tmp_path, monkeypatch):
    search = make_search(tmp_path, max_fe=10)
    interface = mock_interface(search.budget, [[float(20-i)] for i in range(10)])
    monkeypatch.setattr('source.ab_mcts_ahd.InterfaceEC', lambda *a, **k: interface)
    code, filename = search.run()
    assert search.eval_times == 10
    assert json.loads(Path(filename).read_text())['code'] == code
    assert json.loads(Path(filename).read_text())['objective'] == 11
    assert search.mcts.root.visits == 10
    assert sum(1 for call in interface.get_offspring.call_args_list if call.args[1] == 'refine') == 1


def test_initialization_alone_can_exhaust_budget_and_return_best(tmp_path, monkeypatch):
    search = make_search(tmp_path, max_fe=2)
    interface = mock_interface(search.budget, [[8], [5]])
    monkeypatch.setattr('source.ab_mcts_ahd.InterfaceEC', lambda *a, **k: interface)
    _, filename = search.run()
    assert json.loads(Path(filename).read_text())['objective'] == 5
    assert search.eval_times == 2


def test_failed_initialization_attempts_still_fill_n0_within_budget(tmp_path, monkeypatch):
    search = make_search(tmp_path, max_fe=8)
    interface = mock_interface(search.budget, [[float('inf')]] * 3 + [[8], [7], [6], [5], [4]])
    monkeypatch.setattr('source.ab_mcts_ahd.InterfaceEC', lambda *a, **k: interface)
    search.run()
    assert len(search.mcts.root.children) == 5
    assert search.eval_times == 8


def test_e2_is_ineligible_without_an_alternative_parent(monkeypatch):
    tree = AB_MCTS_A('Root', ['test'])
    node = add_node(tree, tree.root, 10, 'e1')
    scored = []
    def score(self, gen, *args):
        scored.append(gen.operator_name)
        return 0
    monkeypatch.setattr(MCTSNode, 'gen_score', score)
    node.hierarchical_thompson_sampling(1, allow_recombination=False)
    assert 'e2' not in scored


@pytest.mark.parametrize('script,expected', [('print(4.0)', [4.0]),
                                           ('import time; time.sleep(10)', 'timeout')])
def test_evaluator_uses_current_interpreter_and_reaps_timeout(tmp_path, monkeypatch, script, expected):
    import subprocess
    import sys
    import time
    from problem_adapter import Problem
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'problems' / 'mock'
    directory.mkdir(parents=True)
    (directory / 'eval.py').write_text(script, encoding='utf-8')
    problem = Problem.__new__(Problem)
    problem.config = SimpleNamespace(timeout=60)
    problem.root_dir = str(tmp_path)
    problem.problem = 'mock'
    problem.problem_type = 'constructive'
    problem.problem_size = 1
    problem.obj_type = 'min'
    problem.output_file = str(directory / 'gpt.py')
    processes = []
    real_popen = subprocess.Popen
    def spawn(args, **kwargs):
        assert args[0] == sys.executable
        process = real_popen(args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr('problem_adapter.subprocess.Popen', spawn)
    start = time.monotonic()
    assert problem.batch_evaluate(['def heuristic(): return 1'], 0, timeout=.5) == expected
    assert time.monotonic() - start < 5
    assert all(process.poll() is not None for process in processes)


def test_agre_stage_chain_and_ablation():
    evol = Evolution.__new__(Evolution)
    evol.agre_stages = {'counterfactual': False}
    seen = []
    def stage(name):
        def call(*args):
            seen.append((name, args[-1]))
            return name
        return call
    for name in ['error_signal', 'counterfactual', 'role_conflict', 'abstraction',
                 'assumption_repair', 'final_advice']:
        setattr(evol, name, stage(name))
    assert evol.ecdrr('candidate') == 'final_advice'
    assert [s[0] for s in seen] == ['error_signal', 'role_conflict', 'abstraction',
                                   'assumption_repair', 'final_advice']
    assert seen[1][1] == 'error_signal'


@pytest.mark.parametrize('problem,expected', [('cvrp_aco', 500), ('tsp_constructive', 1000),
                                             ('kp_constructive', 1000), ('asp_constructive', 1000)])
def test_paper_config_and_adapter_forward_model(tmp_path, problem, expected):
    from ab_mcts_adapter import AB_AHD
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / 'cfg')):
        cfg = compose(config_name='config', overrides=[f'problem={problem}'])
    assert cfg.max_fe == expected
    assert cfg.candidates_per_expansion == 4
    assert cfg.search_timeout == 60
    adapter = AB_AHD(cfg, str(root), tmp_path)
    assert adapter.paras.llm_model_names == ['openrouter/mistralai/codestral-2508']
    assert adapter.paras.ec_operator_weights == [4] * 5


def test_api_uses_requested_model_and_remaining_deadline(monkeypatch):
    from utils import utils
    fake = Mock(return_value=SimpleNamespace(choices=['response']))
    monkeypatch.setattr(utils, 'completion', fake)
    monkeypatch.setattr(utils.time, 'monotonic', lambda: 100.)
    assert utils.chat_completion(1, [], 1., model='requested/model', deadline=120.) == ['response']
    assert fake.call_args.kwargs['model'] == 'requested/model'
    assert fake.call_args.kwargs['timeout'] == 20.
    assert fake.call_args.kwargs['num_retries'] == 0
    with pytest.raises(BudgetExhausted):
        utils.chat_completion(1, [], 1., model='requested/model', deadline=100.)
    assert fake.call_count == 1
