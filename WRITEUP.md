# Stem agent: bounded self-modification with a Pareto stop

Advit Arora.  JetBrains AI Engineering, April 2026.

The brief asks for an agent that, given a class of problems, becomes a
specialist for that class on its own.  The biological framing is in the
prompt: differentiate, but with safeguards that pull bad mutations back.

I read this as a falsifiable claim.  If specialization is real, the same
seed and the same mutator pointed at task classes with different optimal
worker shapes should produce visibly different specialists, *and* each
specialist should be best on its own class compared with the others.  If
neither holds, what I built is a generic self-improver dressed in
stem-cell language.

What follows is the architecture, the experiment, and the things that
went wrong.

## Architecture

### Worker

`agent.py` is a function from `(task, spec, class)` to an output.  The
spec has exactly four fields:

```python
SeedSpec = {
    "system_prompt": "Read the task and return only the requested output.",
    "validation":    "none",        # none | schema | testcases
    "tool_policy":   "no_tools",    # no_tools | validate_retry | code_exec
    "max_retries":   0,             # int in [0, 4]
}
```

`tool_policy` selects the branch.  `no_tools` is one chat completion.
`validate_retry` runs a validator paired with `(validation, class)`,
feeds any failure back, and retries up to `max_retries` times; the
pairings are `("testcases", "regex")` and `("schema", "json")`.
Mismatched pairings (`"schema"` on the regex class, say) fall through
as no-ops, which gives the meta-agent a clean signal next iter that
its mutation didn't engage.  `code_exec` exposes a `python_exec` tool
via OpenAI function calling; the worker can call it up to
`2 + max_retries` times per task.  The sandbox is
`subprocess.run([python, "-I", "-c", code], timeout=6)` with no shell
and no network.  The agent only sees the program's printed output.

The seed is `(none, no_tools, 0)`.  Differentiation means moving to a
different point in
`{none, schema, testcases} × {no_tools, validate_retry, code_exec} ×
[0, 4]` plus an arbitrary system prompt.

### Meta-agent

`evolve.py:_propose()` calls `gpt-4o` (T=0.7) with the task class name,
three demo tasks (with their gold answers), the current spec, and the
per-task feedback from the most recent dev failures.  The reply is
constrained by `response_format={"type": "json_object"}` to:

```
{"reason": "...", "edit": {<field>: <new value>, ...}}
```

The edit is merged into the parent spec.  Temperature was 0.0 in the
first pass; the mutator never proposed a regression, so the rollback
path was uncovered.  T=0.7 fixed that without breaking the structural
constraint.

### Safeguards and the Pareto frontier

A child is rejected on three signals:

1. **Apoptosis (parse)** - extra fields, invalid policy values,
  `max_retries` outside [0, 4], empty system prompt.  Caught before
  any LLM call on the child.
2. **Apoptosis (smoke)** - the agent crashes on a 1-task smoke run, or
  produces an empty output.
3. **Regression rollback** - the child's dev score is strictly below
  the parent's.

Surviving children become candidate points
`P = (dev_score, mean_in_tokens_per_eval)`.  The accepted set is the
Pareto frontier under the partial order
`A ≽ B  iff  A.score ≥ B.score  and  A.tokens ≤ B.tokens`, with at
least one strict.  New points that aren't dominated are added; existing
points the new one dominates are removed
(`evolve.py:update_frontier`).  The reported specialist is the
highest-scoring frontier entry, ties broken by lower tokens.

The stopping criterion counts consecutive iterations that fail to add
a frontier point.  After three in a row the loop reports a *plateau*.
With `iters=8` this fires often before the cap.  The cap is the safety
net.

`tools/safeguards_test.py` injects pathological children to exercise
every rejection path directly, since natural runs do not always trigger
`parse` or `smoke`.

## Three task classes

I picked classes where the optimal worker shape is, by hypothesis, a
different point in the spec space.  If the meta-agent finds the same
mutation everywhere, the differentiation claim collapses to "the
meta-agent always proposes the same good thing".

- **regex synthesis** - 39 hand-curated tasks.  Input is a
  description plus positives and negatives; output is a regex; the
  validator compiles it and runs it.  Predicted optimum:
  `(testcases, validate_retry, max_retries ≥ 2)`.  The validator's
  feedback ("should match X but didn't") gives the worker enough
  information to anchor or tighten the pattern on retry.
- **bug-report metadata extraction** - 39 hand-curated GitHub-style
  items.  Output is `{severity, kind, area}` from a closed schema;
  most failures are *value* errors (severity = high vs critical), not
  schema errors.  Predicted optimum: a system-prompt rewrite that
  embeds rules and worked patterns, because the schema validator
  cannot recognise wrong-but-legal answers.
- **arithmetic word problems** - 42 hand-curated items.
  `gpt-4o-mini`'s arithmetic compounds errors past three steps.
  Predicted optimum: `code_exec` plus a system prompt that tells the
  worker to reach for `python_exec`.

Each class has a hand-picked split: 6 demo, 12 dev, the rest (21-24)
test.  The dev set is over-weighted with items the seed fails on.
JSON dev was rebalanced once after a first run because the meta-agent
trained on critical-heavy dev failures and learned "everything is
critical", which then hurt the test score; v2 dev mixes low and med
items with the harder ones.

## Results

Two seeds per class, plain mean.  Total OpenAI cost for the sweep was
roughly $1.

| class | seed test (mean of 2) | specialist test | Δ | specialist `tool_policy` | specialist `validation` | system_prompt edit |
|---|---:|---:|---:|---|---|---|
| regex | 69% (67% / 71%) | 84% (81% / 86%) | **+15** | `validate_retry` | `testcases` | regex-specific rule in s0, unchanged in s1 |
| json  | 43%             | 48%             | **+5**  | mixed (no_tools / validate_retry) | mixed (none / schema) | rules + worked patterns |
| math  | 62%             | 79%             | **+17** | `code_exec` | `none` | "use python_exec for calculations" |

Same seed, same mutator, three different specialist shapes.  The regex
specialist enables the testcase validator and a small retry budget.
The math specialist switches to `code_exec` and adds a one-line nudge
to the system prompt.  The json specialist's most consistent edit is a
prompt rewrite that adds severity definitions and worked patterns; in
one of the two seeds it also turned on the schema validator, but
schema-validation cannot recognise wrong-category errors so the gain
on test was marginal.

### Cross-class transfer (test sets)

| spec | regex | json | math |
|---|---:|---:|---:|
| seed | 71% | 43% | 67% |
| regex specialist | **86%** | 38% | 62% |
| json specialist | 62% | **48%** | 62% |
| math specialist | 67% | 43% | **71%** |

The diagonal is the best entry in every row.  Every off-diagonal is
flat or below seed.  The largest drop is the json specialist on
regex (-9 pp): the long category-rules system prompt actively
confuses a regex worker that just needs a pattern.  The regex
specialist falls 5 pp on json and 5 pp on math because its prompt
edit ("ensure the regex meets all conditions") is regex-specific
boilerplate that adds noise to other classes.  The math specialist's
`python_exec` mention is ignored on the other classes' tasks but its
prompt edit shaves 4 pp off regex.  These drops are small in
absolute terms but they go the way they should: each spec is shaped
to its class.

### A lineage that actually used the safeguards

Math, seed 1 (`runs/math/seed1.json`):

```
iter 0  seed                                                  dev 3/12 = 25%
iter 1  edit={tool_policy:code_exec, max_retries:2,
              system_prompt:"... use python_exec ..."}         dev 8/12 = 67%   FRONTIER+
iter 2  edit={max_retries:3}                                   dev 7/12 = 58%   ROLLBACK
iter 3  edit={max_retries:3}                                   dev 7/12 = 58%   ROLLBACK
iter 4  edit={max_retries:3}                                   dev 7/12 = 58%   ROLLBACK
        plateau: 3 non-improving iters in a row, stop.
TEST: seed 15/24 = 62%   specialist 19/24 = 79%
```

Iter 1's edit bundles the right tool, the right tool budget, and a
system-prompt nudge in one mutation.  The meta-agent's reasoning
field for that proposal was "the worker is failing tasks that
require precise arithmetic and multi-step reasoning"; the resulting
spec hits 67 % on dev and 79 % on test.  After that the meta-agent
keeps proposing higher `max_retries` looking for one more dev-task
fix; all three children regress (the extra turns cost more than they
help) and get rolled back.  In an earlier seed=0 run the meta-agent
proposed `max_retries: 6`, which `parse_check` rejected as
out-of-range without consuming an LLM call on the worker.

## What surprised me

**The meta-agent only enabled `code_exec` properly once I told it
about the coupling between tool and prompt.**  In the first version,
when the meta-agent proposed `tool_policy: code_exec` it left
`system_prompt` alone.  The worker *could* call `python_exec` but
often didn't, and on test the math specialist scored *below* seed
(50 % vs 62 %).  After I added one sentence to the meta-agent's
system prompt - "when you turn on code_exec, also edit system_prompt
so the worker actually uses python_exec" - the meta-agent started
proposing both edits in the same iter and the math specialist beat
seed by 12-17 pp across two seeds.  This is the most fragile
behaviour I observed: the worker's tool-availability and its
tool-using habit are coupled, but the spec representation treats
them as independent.  Without a hint the meta-agent missed that
coupling.

**JSON underperformed and rebalancing the dev split helped only a
little.**  My first JSON dev set was all critical-and-regression
items the seed failed on.  The meta-agent learned "tag everything
as critical regression" and dev hit 100 % while test stayed at 52 %.
Rebalancing dev to 4 critical, 2 high, 5 low, 1 med, with mixed
kinds, brought specialists down to 48 %, which is +5 pp over the
seed's 43 % - real but marginal.  My read: classes whose failures
need *judgement* (rather than parse-correctness or arithmetic-
correctness) are genuinely hard for this evolution loop, because
the validator can only see structural problems, and the meta-agent
ends up doing in-context-learning prompt engineering on a tiny
demo+dev set that does not generalise well.

## What didn't work

**Strict-improvement acceptance.**  First version of the rule: child
accepted iff dev score strictly greater than parent.  With a 12-task
dev set, mutations that fix a *test* item without moving the *dev*
score get thrown away.  The regex run produced no specialist at all
under that rule.  I switched to monotone (`≥`) plus the Pareto check,
and added the plateau counter so the loop still terminates when the
meta-agent stops finding new Pareto points.

**The validator dispatch was a bug.**  I keyed validators on task
class (`{"regex": ..., "json": ...}`) but the spec uses policy names
(`testcases`, `schema`).  For about an hour the meta-agent's
proposals to enable validation were silent no-ops and dev never
moved.  Fixed by pairing `(policy, class)` explicitly.

**Backtick stripping was bigger than expected.**  `gpt-4o-mini` wraps
regex output in single backticks (` `^abc$` `).  My validator
compiled the literal string with backticks and rejected it as
invalid regex.  Stripping a single-backtick wrap moved the regex
seed baseline from ~17 % to 71 %.  I almost shipped with the wrong
baseline.

**The rollback ablation didn't dramatically widen.**  Running math
with `--no-rollback` produced a similar test score (within ±5 pp) to
the with-rollback run.  Parse-apoptosis still fires (out-of-range
`max_retries`) and the meta-agent's proposals are conservative
enough that natural regressions are infrequent.  I would expect a
larger gap with a bigger iteration budget and a hotter meta-agent.

## What I would do next, with another week

1. **Bigger dev set, around 30 items.**  Twelve is small enough that
   one task flipping shifts the dev score by 8 pp, which is right on
   top of the gpt-4o-mini noise floor.  Strict-improvement
   acceptance might actually work at that size.
2. **Tagged failure breakdown for the JSON eval.**  Right now the
   meta-agent sees "wrong" and has to guess whether the issue is
   schema or value.  Splitting "schema-fail" / "value-mismatch" /
   "missing-field" should let it propose targeted edits - and would
   stop it from over-confidently rewriting the prompt to say
   "everything is critical" when the dev failures happen to be
   biased.
3. **A fourth task class with a *different* tool: SQL from natural
   language, against a small sqlite database.**  Same `code_exec`
   pattern but a different sandboxed tool.  I'd want to see whether
   the meta-agent discovers the SQL tool the way it discovered
   `code_exec` for math, or whether the discovery only generalises
   along the existing tool path.
4. **Run multiple proposals per iter and let Pareto pick the best,
   instead of one proposal per iter.**  This changes the search from
   greedy to beam-2.  ADAS does something like this with an archive
   of candidates; my budget couldn't afford it but the architecture
   supports it.

## References

- ADAS - Hu, Lu, Clune.  *Automated Design of Agentic Systems*.  ICLR 2025.  arXiv:2408.08435.
- DGM - Sakana AI.  *Darwin Gödel Machine*.  2025.  arXiv:2505.22954.
- GEPA - Agrawal et al.  *Reflective prompt evolution*.  2025.  arXiv:2507.19457.
- Voyager - Wang et al.  *Skill library and curriculum in MineDojo*.  2023.  arXiv:2305.16291.
- Anthropic.  *Building effective agents*.  2024.  Canonical patterns: prompt chaining, routing, parallelisation, orchestrator-subagent, evaluator-optimizer.
