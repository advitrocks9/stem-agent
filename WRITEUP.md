# Stem agent: bounded self-modification under monotone acceptance

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

### Safeguards and acceptance

A child is rejected on three signals:

1. **Apoptosis (parse)** - extra fields, invalid policy values,
  `max_retries` outside [0, 4], empty system prompt.  Caught before
  any LLM call on the child.
2. **Apoptosis (smoke)** - the agent crashes on a 1-task smoke run, or
  produces an empty output.
3. **Regression rollback** - the child's dev score is strictly below
  the parent's.

Surviving children become points `P = (dev_score, mean in_tokens per
eval)` in an accepted-archive.  I implement Pareto dominance over
those points (`A ≽ B iff A.score ≥ B.score and A.tokens ≤ B.tokens`,
strict in at least one) and reject children that any existing point
already dominates; the reported specialist is the highest-scoring
archive entry, ties broken by lower tokens.  In practice on this
problem the score-cost tradeoff doesn't bite: in every accepted
trajectory the highest-scoring point is also the cheapest, so the
dominance check reduces to "monotone in score plus a tie-breaker on
tokens".  The Pareto branding earns its keep only as the
machinery that powers a clean stop signal: a plateau, defined as
three iterations in a row that fail to add a non-dominated point.

The stopping criterion is the consecutive-no-new-point count.  After
three in a row the loop terminates as a plateau.  With `iters=8`
this fires often before the cap.

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

Three seeds per class, mean ± standard deviation across seeds.  Total
OpenAI cost was about $1.50 across the nine evolution runs plus the
transfer matrix.

| class | seed test (n=3) | specialist test (n=3) | Δ mean ± SD | per-seed Δ | specialist's spec converges to |
|---|---:|---:|---:|---|---|
| regex | 68% ± 2     | 84% ± 3      | **+16 ± 5.5** | +10, +19, +19 | `validate_retry`, `testcases`, `max_retries` 2-4 |
| json  | 44% ± 3     | 49% ± 7      | **+5 ± 9.5**  | +14, -5, +5  | system_prompt rewrite, validation policy mixed |
| math  | 62% ± 0     | 75% ± 7      | **+13 ± 7.2** | +17, +17, +5 | `code_exec` + "use python_exec" prompt nudge |

Same seed, same mutator, three different specialist shapes.  The regex
specialist enables the testcase validator and a small retry budget.
The math specialist switches to `code_exec` and adds a one-line nudge.
The json specialist's most consistent edit is a system-prompt rewrite
with severity definitions and worked patterns.  The shapes are stable
across seeds even when the magnitudes are not.

The JSON variance is the headline weakness.  +14 / -5 / +5 across
seeds means that on at least one seed the meta-agent's prompt rewrite
overfit dev (which hit 83% on s1) and underperformed seed on test by
5 pp.  The class is genuinely harder than the other two: the dev set
has 12 items and the test set 21, so a prompt rule that fits the
dev distribution will not necessarily fit the test distribution
unless dev is large enough to constrain the meta-agent's edits.
This is not a bug in the loop; it is a sample-size limit on how
much a 12-task dev split can be trusted as a signal for 21-task
generalisation.

I added typed feedback to the JSON eval halfway through (commit
log on `2026-05-01`): instead of "wrong" the meta-agent now sees
`[parse-fail]`, `[missing-field]`, `[value-not-allowed]`, or
`[value-mismatch]` per task.  This is the exact distinction the
meta-agent needs to decide whether `validate_retry` will help (it
does for the first three) or whether only a system-prompt edit
will (the fourth).  The mean delta moved from +5 (untyped) to +5
(typed) which is a wash on the average, but the *variance* widened
because the meta-agent now commits more confidently to the prompt-
rewrite path; two of three seeds beat seed by 5+ pp, the third
overfit and lost 5 pp.  Typed feedback gives the meta-agent a
sharper instrument; it does not give it a bigger training set.

### Cross-class transfer (test sets)

Each cell is the best-of-3-seeds specialist on the named class's test set.

| spec | regex | json | math |
|---|---:|---:|---:|
| seed | 71% | 43% | 62% |
| regex specialist | **81%** | 48% | 62% |
| json specialist | 67% | **57%** | 67% |
| math specialist | 71% | 38% | **83%** |

The diagonal is the best entry in every row.  Off-diagonal cells
mostly fall back toward seed; the math specialist drops 5 pp on json
because its `code_exec`+python prompt is irrelevant there and
clutters the worker's input.  The json specialist on regex drops
4 pp.  The two off-diagonal cells where a specialist beats seed
(json spec on math, +5 pp; regex spec on json, +5 pp) are inside
the per-task noise floor of these test sets and do not look
intentional.  The strong specialization signal is the diagonal
beating its column mean, not the off-diagonals collapsing.
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
(50% vs 62%).  After I added one sentence to the meta-agent's system
prompt asking it to edit `system_prompt` whenever it changed
`tool_policy`, the meta-agent started proposing both edits in the
same iter and math beat seed by 5-17 pp across three seeds.  This is
the most fragile behaviour I observed: the worker's tool-availability
and its tool-using habit are coupled in the model's behaviour, but
the spec representation treats them as independent fields.

**`max_retries` is a different knob in different branches.**  Under
`validate_retry` it caps re-prompts; under `code_exec` (after the
patch) it caps tool-call rounds; under `no_tools` it does nothing.
The meta-agent treats it as a single number anyway, which works only
because each tool_policy uses it sensibly.  It would not survive a
fourth branch added without thought.

**The Pareto frontier is overclaimed.**  I called the accepted-archive a
"Pareto frontier" because that's what the dominance rule technically is.
In this problem it never produces a real score-cost tradeoff: the
highest-scoring archive entry is also the cheapest in every accepted
trajectory.  So the structure I shipped reduces to monotone-in-score
with a token-tiebreaker.  The dominance machinery is still doing one
useful thing: it lets dominated children count toward the patience
budget, so the loop stops promptly when the meta-agent has run out of
ideas.  But "Pareto frontier" reads like I'm claiming I built a
multi-objective search and the reader should expect a tradeoff curve.
I'm not, and they shouldn't.

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
under that rule.  I switched to monotone (`≥`) plus the dominance
check on tokens, and added the plateau counter so the loop still
terminates when the meta-agent stops finding new archive points.

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
4. **Run multiple proposals per iter and pick the best by archive
   admission, instead of one proposal per iter.**  This changes the
   search from greedy to beam-2.  ADAS does something similar with an
   archive of candidates; my budget couldn't afford it but the
   architecture supports it.

## References

- ADAS - Hu, Lu, Clune.  *Automated Design of Agentic Systems*.  ICLR 2025.  arXiv:2408.08435.
- DGM - Sakana AI.  *Darwin Gödel Machine*.  2025.  arXiv:2505.22954.
- GEPA - Agrawal et al.  *Reflective prompt evolution*.  2025.  arXiv:2507.19457.
- Voyager - Wang et al.  *Skill library and curriculum in MineDojo*.  2023.  arXiv:2305.16291.
- Anthropic.  *Building effective agents*.  2024.  Canonical patterns: prompt chaining, routing, parallelisation, orchestrator-subagent, evaluator-optimizer.
