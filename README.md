# GENESIS

Run from the repository root after installing `requirements.txt`. The default
model is `openrouter/mistralai/codestral-2508`, using OpenRouter (`OPENROUTER_API_KEY`).
The configured `model` is passed through to the API without a hard-coded override.

```bash
python main.py
python main.py problem=tsp_constructive seed=2025
python main.py model=openrouter/mistralai/codestral-2508
```

On Windows, `py -3.11` can replace `python`. Evaluation subprocesses use the
same Python interpreter as the parent process.

## Paper settings and budgets

- Initial pool: 5 heuristics total.
- Expansion: 4 candidates followed by at most one AGRE refinement of the best
  valid candidate. Invalid candidate evaluations can be retried, within budget.
- FE limit: 500 by default; 1,000 for TSP/KP/ASP constructive configurations.
  Every evaluator invocation, including failed evaluations and refinement,
  consumes one FE. Generation-only calls do not consume FE.
- `search_timeout=60`: elapsed seconds for the search, including initialization,
  generation, reflection and evaluation. Remaining time is passed to API requests
  and evaluator processes. No new work starts after the deadline; request
  cancellation and final checkpoint writing may add a small shutdown overhead.
- `timeout=60`: separate timeout for a single heuristic evaluation, capped by the
  remaining search time. Final validation in `main.py` is outside the search budget.
- A partial last batch is retained when a budget expires. If initialization yields
  no valid heuristic before expiry, the run reports this explicitly.

To run with only an FE limit, explicitly use `search_timeout=null`. This changes
the paper's stated wall-clock setting. Use different `seed` values for independent
runs; the seed controls Python/NumPy search randomness, not external LLM sampling.

## Search conventions

Real-node posteriors contain their own evaluation and descendant evaluations.
Operator posteriors receive observations along the corresponding generation path.
Each observation updates the shared global NIG posterior **once**, regardless of
path length. Local operator priors stay fixed; global evidence is combined with
local samples during HTS rather than being repeatedly reused as a local prior.

Two boundary cases need explicit conventions because a virtual root has no code
and S1 needs multiple heuristics: the root uses E1 for diversity generation and
compares it against continuation into its real children; real nodes use
`counter/e2/m1/m2/s1`, with S1 eligible only when the path contains at least two
real nodes and E2 requiring an alternative parent in the elite pool. There is no
fixed child-count cap or depth-based random fallback.
These boundary conventions should be made explicit in the manuscript's algorithm.

The elite pool is deduplicated by code. Equal descriptions or objective values do
not discard distinct evaluated heuristics from the tree or posterior evidence.

## Ablations

```bash
python main.py ablation.posterior_sampling=false
python main.py ablation.adaptive_op_selection=false
python main.py ablation.global_sharing=false
python main.py ablation.agre=false
python main.py ablation.agre_stages.counterfactual=false
```

Without adaptive operator selection, expansion versus continuation still uses
the best GEN score, then the operator is chosen uniformly among eligible actions.
Without global sharing, the global prior remains fixed and GEN scoring uses only
local information. Disabling an AGRE stage passes its input to the next stage.

## Offline verification

Install `pytest` in the development environment, then run:

```bash
python -m pytest tests/test_genesis.py -q
```

These regressions exercise posterior arithmetic, selection, model configuration,
AGRE, FE/deadline handling and a complete mocked search without paid API calls or
benchmark dataset changes. They do not reproduce the manuscript's benchmark results.
