# stem-agent

A small experiment in self-specializing agents. The seed is a single ReAct
worker with a four-field spec; a meta-agent proposes structured edits to the
spec; surviving children are admitted to an accepted-archive with a
Pareto-dominance rejection rule on (dev score, inference cost). The
score-cost tradeoff turns out not to bite for this problem -- in every
accepted trajectory the best-scoring archive entry is also the cheapest --
so the dominance check effectively reduces to monotone-in-score plus a
plateau stop. Same seed and same mutator on three task classes produce
three differently-shaped specialists.

## Setup

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run

```bash
uv run python stem.py --class regex --iters 8                # one class, one seed
uv run python tools/run_all.py                               # 3 classes x 2 seeds
uv run python tools/transfer.py                              # 4x3 transfer matrix
uv run python tools/safeguards_test.py                       # rejection-path checks
uv run python tools/print_lineage.py runs/regex/seed0.json   # ASCII trace
```

The full multi-seed sweep is roughly $1 in OpenAI tokens (gpt-4o-mini for
the worker, gpt-4o for the meta-agent) and finishes in about thirty
minutes on a residential connection.

## Layout

`agent.py` - worker. `(task, spec, class) → output`. Three branches by tool
policy. The validator dispatch is keyed on `(policy, class)` pairs; mismatches
fall through.

`evolve.py` - meta loop. The meta-agent proposes a JSON edit over four spec
fields. Surviving children are admitted to an archive under Pareto dominance
on (score, mean in-tokens per eval). The loop stops after three iterations
in a row that fail to add a non-dominated point, or at the iter cap.

`eval.py` - task loader, splits, scoring. Math answers are normalised
(strip currency symbols, prefer the last number, accept HH:MM and a/b
fractions). JSON answers are scored on field-by-field equality with the
gold mapping.

`llm.py` - thin OpenAI wrapper; one retry on connection errors.

`tasks/{regex, json, math}/tasks.json` - 39 + 39 + 42 hand-curated items.

`tools/` - the analysis scripts above.

## Headline numbers

See `WRITEUP.md` for the full table and the lineage walk-through. In short:
each class produced a specialist whose spec differs from the seed on at
least two fields; the regex specialist enables the testcase validator with
a small retry budget, the math specialist switches to `code_exec`, and the
json specialist rewrites the system prompt.

## Things I'd tell a teammate before they extend this

- The dev set per class is twelve tasks. That's small enough for a single
  task flip to shift dev score by eight percentage points, which is right
  on top of the gpt-4o-mini stochasticity floor. I would bump to thirty
  before trusting strict-improvement acceptance over the monotone rule.
- The validator falls through silently on mismatched (policy, class)
  pairings. That's deliberate: it lets the meta-agent learn from the dev
  failures rather than from a thrown exception. If you log inside the
  worker, do not log at WARN for this case.
- `code_exec` uses `subprocess.run([python, "-I", ...], timeout=6)`. No
  network sandbox, just the timeout and `-I`. If you point this at
  untrusted code, harden the sandbox first.
