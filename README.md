# stem-agent

A small experiment in self-specializing agents. The seed is one ReAct
worker with a four-field spec; a meta-agent proposes structured edits;
surviving children are admitted to a Pareto-dominance archive on
(dev_score, mean prompt tokens). Same seed and same mutator on four task
classes (regex synthesis, bug-report metadata extraction, arithmetic word
problems, SQL from natural language) produce four differently-shaped
specialists.

## Setup

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run

```bash
uv run python stem.py --class regex --iters 8                # one class, one seed
uv run python tools/run_all.py                               # 4 classes x 3 seeds
uv run python tools/transfer.py                              # 5x4 transfer matrix
uv run python tools/safeguards_test.py                       # rejection-path checks
uv run python tools/split_sensitivity.py                     # 5 alt splits x 3 classes
uv run python tools/run_ablations.py                         # random / no-demos / no-feedback
uv run python tools/bootstrap_ci.py                          # paired test-item bootstrap
uv run python tools/hand_baseline.py                         # 10-min hand-tuned spec per class
uv run python tools/recompute_headline.py                    # source-of-truth headline table
uv run python tools/plots.py                                 # trajectories + transfer + Pareto
uv run python tools/print_lineage.py runs/regex/seed0.json   # ASCII trace
```

The full sweep is about $4 in OpenAI tokens (gpt-4o-mini for the worker,
gpt-4o for the meta-agent) and finishes in about two hours.

## Layout

`agent.py` - worker. `(task, spec, class) -> output`. Three branches by
tool policy. The validator dispatch is keyed on `(policy, class)` pairs;
mismatches fall through.

`evolve.py` - meta loop. The meta-agent proposes a JSON edit over four
spec fields. Surviving children are admitted to an archive under Pareto
dominance on (score, mean prompt tokens). The loop stops after three
iterations in a row that fail to add a non-dominated point, or at the
iter cap.

`eval.py` - task loader, splits, scoring. Math answers are normalised
(prefer the number adjacent to the word "answer", then the number in the
last sentence; strip currency symbols; accept HH:MM and a/b fractions).
JSON answers are scored on field-by-field equality with the gold mapping.

`llm.py` - thin OpenAI wrapper; one retry on connection errors.

`tasks/{regex, json, math, sql}/tasks.json` - 39 + 39 + 42 + 30
hand-curated items. The SQL class also ships a sqlite fixture
(`tasks/sql/fixture.sql`) that the validator runs queries against.

`tools/` - the analysis scripts above.

`tests/` - unit tests for the math answer normalizer (the verbose-output
case bit me on math seed 2 evaluation).

## Headline numbers

Source: `runs/{class}/seed{0,1,2}.json`. Reproduced by
`tools/recompute_headline.py`.

| class | seed test | hand strong | specialist | Δ vs seed |
|---|---:|---:|---:|---:|
| regex | 71% | 76% | 83% ± 3 | **+11 ± 2.7** |
| json  | 43% | 48% | 49% ± 7 | **+6 ± 7.3** |
| math  | 62% | 100% | 75% ± 8 | **+12 ± 8.3** |
| sql   | 92% | 100% | 97% ± 5 | **+6 ± 4.8** |

The hand-strong column is a 10-minute hand-written spec scored on the
same test set, so the meta-agent's gain is measured against a real
baseline. The headline finding: the meta-agent beats the hand spec on
regex and ties on sql; on math it loses by 25pp, because the hand prompt
is more directive about when to call `python_exec` than the meta-agent's
typical one-line nudge. See `WRITEUP.md` for the full story.

## Things I'd tell a teammate before they extend this

- The dev set per class is twelve tasks. That's small enough for a
  single task flip to shift dev score by eight percentage points, which
  is right on top of the gpt-4o-mini stochasticity floor. I would bump
  to thirty before trusting strict-improvement acceptance over the
  monotone rule.
- The validator falls through silently on mismatched (policy, class)
  pairings. That's deliberate: it lets the meta-agent learn from the dev
  failures rather than from a thrown exception. If you log inside the
  worker, do not log at WARN for this case.
- `code_exec` uses `subprocess.run([python, "-I", ...], timeout=6)`. No
  network sandbox, just the timeout and `-I`. If you point this at
  untrusted code, harden the sandbox first.
- The math answer normalizer used to take the last number in the string;
  verbose worker outputs broke that. It now anchors on the word
  "answer". `tests/test_eval_norm.py` covers the regression cases.
