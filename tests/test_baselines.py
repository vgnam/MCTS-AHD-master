"""Offline tests for the MoH and HiFo-Prompt baselines (mocked LLM and evaluator)."""

import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from problem_adapter import Prompts
from source import pop_greedy, prob_rank
from source.getParas import Paras
from source.hifo_prompt import EvolutionaryNavigator, HiFo_AHD, InsightPool
from source.moh import SEED_OPTIMIZER, MoH_AHD, Pop

ROOT = Path(__file__).resolve().parents[1]


def compose_cfg(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'cfg')):
        return compose(config_name='config', overrides=list(overrides))


class FakeProblem:
    """Objective = the integer assigned to ``v`` in the heuristic (invalid if absent)."""

    def __init__(self, cfg):
        self.problem = cfg.problem.problem_name
        self.problem_size = cfg.problem.problem_size
        self.prompts = Prompts(cfg.problem, str(ROOT))
        self.calls = 0

    def batch_evaluate(self, codes, iteration, timeout=None):
        self.calls += 1
        match = re.search(r"v = (\d+)", codes[0])
        return [float(match.group(1))] if match else [float('inf')]


class FakeLLM:
    """Answers by prompt type; heuristic objectives decrease over time."""

    def __init__(self, improver_code=SEED_OPTIMIZER):
        self.counter = itertools.count()
        self.improver_code = improver_code
        self.prompts = []

    def __call__(self, model, messages, temperature, n, timeout, num_retries):
        text = messages[-1]['content']
        self.prompts.append(text)
        if 'extract 1-2 concise' in text:
            reply = '- Combine distance and demand features with adaptive weights for each step.'
        elif 'number of iterations the outer loop' in text:
            reply = 'Analysis.\n```txt\n2\n```'
        elif 'for generating the seed prompt' in text:
            reply = '```json\n{"direction": [{"content": "greedy"}, {"content": "regret"}]}\n```'
        elif text.lstrip().startswith('Improve the following solution') and 'improve_algorithm' in text:
            version = f'\nOPTIMIZER_VERSION = {next(self.counter)}'  # distinct code per reply
            reply = '{use a fresh random parent}\n```python\n' + self.improver_code + version + '\n```'
        elif text.lstrip().startswith('Given the following heuristic'):
            reply = '```json\n{"insights": ["refine weights", "try lookahead"]}\n```'
        else:
            k = next(self.counter)
            reply = (f'{{idea {k}}}\n```python\nimport numpy as np\ndef heuristic(x):\n'
                     f'    v = {1000 - k}\n    return v\n```')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def make_paras(tmp_path, max_fe):
    paras = Paras()
    paras.set_paras(init_size=3, ec_fe_max=max_fe, llm_model='test-model', pop_size=10,
                    exp_output_path=str(tmp_path))
    return paras


@pytest.fixture
def cfg():
    return compose_cfg('problem=tsp_constructive', 'search_timeout=null')


def test_hifo_spends_exact_budget_and_uses_hindsight_and_foresight(tmp_path, cfg, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr('utils.utils.completion', llm)
    problem = FakeProblem(cfg)
    search = HiFo_AHD(make_paras(tmp_path, 40), problem, prob_rank, pop_greedy, cfg=cfg)
    code, filename = search.run()

    assert search.eval_times == problem.calls == 40
    best = json.loads(Path(filename).read_text())
    assert best['code'] == code and best['objective'] == min(i['objective'] for i in search.archive.items)
    assert any('Combine distance and demand' in tip for tip in search.insight_pool.tips)
    assert any("Consider these successful design principles" in p for p in llm.prompts)
    assert any("please pay special attention to" in p for p in llm.prompts)
    assert (tmp_path / 'hifo_prompt_log.json').exists()


def test_hifo_ablation_switches_remove_insights_and_directives(tmp_path, cfg, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr('utils.utils.completion', llm)
    cfg.baselines.hifo.use_insights = False
    cfg.baselines.hifo.use_navigator = False
    search = HiFo_AHD(make_paras(tmp_path, 20), FakeProblem(cfg), prob_rank, pop_greedy, cfg=cfg)
    search.run()
    assert not any("successful design principles" in p for p in llm.prompts)
    assert not any("pay special attention to" in p for p in llm.prompts)
    assert not any('extract 1-2 concise' in p for p in llm.prompts)


def test_navigator_switches_to_exploration_on_stagnation():
    nav = EvolutionaryNavigator()
    history = [5.0, 5.0]
    regimes = [nav.get_guidance(history, [1.0])[0] for _ in range(4)]
    assert regimes[-1] == 'exploration'


def test_insight_pool_dedups_and_evicts_least_effective():
    pool = InsightPool(max_size=6)
    assert not pool.add_tip(pool.DEFAULT_TIPS[0])  # near duplicate
    tip = 'Prefer nearest feasible customers when residual capacity is tight'
    assert pool.add_tip(tip)
    for t in list(pool.tips):
        pool.tip_stats[t]['used_count'] = 5
    pool.update_tip_stats(tip, -1.0)
    pool.add_tip('Use lookahead over two candidate moves before committing to a choice')
    assert tip not in pool.tips


def test_moh_spends_exact_budget_and_improves_meta_population(tmp_path, cfg, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr('utils.utils.completion', llm)
    problem = FakeProblem(cfg)
    search = MoH_AHD(make_paras(tmp_path, 30), problem, cfg=cfg)
    code, filename = search.run()

    assert search.eval_times == problem.calls == 30
    assert json.loads(Path(filename).read_text())['code'] == code
    assert search.improver_pop.get_subtask_size('meta-optimizer') >= 2
    state = json.loads((tmp_path / 'moh_state.json').read_text())
    assert state['log'] and state['heuristics']


def test_moh_budget_stop_escapes_generated_try_except(tmp_path, cfg, monkeypatch):
    swallowing = SEED_OPTIMIZER.replace(
        "    solutions_with_utilities = [(idea, solution, utility(solution, idea, task)) for idea, solution in zip(new_ideas, new_solutions)]",
        "    solutions_with_utilities = []\n"
        "    for idea, solution in zip(new_ideas, new_solutions):\n"
        "        try:\n"
        "            solutions_with_utilities.append((idea, solution, utility(solution, idea, task)))\n"
        "        except Exception:\n"
        "            solutions_with_utilities.append((idea, solution, 1e10))")
    assert swallowing != SEED_OPTIMIZER
    monkeypatch.setattr('utils.utils.completion', FakeLLM(improver_code=swallowing))
    problem = FakeProblem(cfg)
    search = MoH_AHD(make_paras(tmp_path, 12), problem, cfg=cfg)
    search.run()
    assert search.eval_times == problem.calls == 12


def test_moh_population_keeps_best_and_rejects_non_code():
    pop = Pop(['t'], 2)
    assert not pop.save_solution('t', 'idea', 'x = 1', 1.0)
    for value in (3.0, 1.0, 2.0):
        pop.save_solution('t', 'idea', f'def f():\n    return {value}', value)
    assert [e['utility'] for e in pop.get_population('t')] == [1.0, 2.0]


@pytest.mark.parametrize('algorithm,adapter', [('moh', 'MoH_AHD_Adapter'),
                                               ('hifo-prompt', 'HiFo_AHD_Adapter')])
def test_config_and_adapters(tmp_path, algorithm, adapter):
    import baseline_adapters
    cfg = compose_cfg('problem=cvrp_aco', f'algorithm={algorithm}')
    assert cfg.baselines.moh.seed_pop_size == cfg.init_pop_size
    lhh = getattr(baseline_adapters, adapter)(cfg, str(ROOT), tmp_path)
    assert lhh.paras.llm_model == cfg.model
    assert lhh.paras.ec_fe_max == 500
    assert type(lhh.build()).__name__ in ('MoH_AHD', 'HiFo_AHD')
