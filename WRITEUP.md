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

## Four task classes

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
- **SQL from natural language** - 30 hand-written tasks against a
  small sqlite fixture (3 tables, 9 users, 14 products, 25 orders).
  The schema is prepended to the user message in the worker (Spider/
  BIRD convention) so the seed isn't class-blind.  Predicted optimum:
  `(results, validate_retry, max_retries ≥ 1)` - the validator runs
  the predicted SQL against the fixture and feeds back row-count or
  result-mismatch failures, which is enough information to fix
  off-by-one joins on a single retry.

Each class has a hand-picked split: 6 demo, 12 dev, the rest (21-24)
test.  The dev set is over-weighted with items the seed fails on.
JSON dev was rebalanced once after a first run because the meta-agent
trained on critical-heavy dev failures and learned "everything is
critical", which then hurt the test score; v2 dev mixes low and med
items with the harder ones.

## Results

Three seeds per class, mean ± standard deviation across seeds.  Total
OpenAI cost across all the runs in this writeup (12 main evolution
runs, 5 split-sensitivity sweeps × 3 classes, the cross-class
transfer matrix, and the ablations described below) was about $4.

Numbers are paired re-evaluations of seed and specialist on the same
test items, so every Δ is a within-task comparison; bootstrap 95% CIs
are computed by resampling the test items 1000 times per seed.

| class | seed test | specialist test | Δ mean | per-seed Δ | best per-seed bootstrap 95% CI | specialist spec converges to |
|---|---:|---:|---:|---|---|---|
| regex | 70% ± 3 | 83% ± 3  | **+13** | +9, +19, +9 | `[+5, +38]` (seed 1) | `validate_retry`, `testcases`, `max_retries` 2-4 |
| json  | 48% ± 0 | 49% ± 7  | **+2**  | +9, -5, +0  | `[-9, +29]` (seed 0)  | system_prompt rewrite, validation policy mixed |
| math  | 64% ± 2 | 72% ± 5  | **+8**  | +9, +12, +4 | `[-8, +33]` (seed 1)  | `code_exec` + "use python_exec" prompt nudge |
| sql   | 92% ± 0 | 100% ± 0 | **+8**  | +8, +8, +8  | `[+0, +25]` (any)     | `validate_retry`, `results`, `max_retries=2` |

The honest read: regex is the only class whose strongest seed has a
positive-only CI; on the others the per-seed CIs straddle zero
because the test sets are 12-24 tasks, which is small for a paired
proportion test.  The cross-seed mean is what gives the differentiation
claim its weight, not the per-seed CI on any one run.  Larger test
sets would tighten these.  See `runs/bootstrap_summary.json`.

Same seed, same mutator, four different specialist shapes.  The regex
specialist enables the testcase validator and a small retry budget.
The math specialist switches to `code_exec` and adds a one-line nudge.
The json specialist's most consistent edit is a system-prompt rewrite
with severity definitions and worked patterns.  The SQL specialist
enables the results validator with a single retry; the seed already
sits at 92% on this class because the schema-injection convention
gives it most of what it needs, and the validator picks up the one
test item the seed gets wrong.  The SQL convergence is the cleanest
of the four: three independent meta-seeds end at the same spec.

The shapes are stable across seeds even when the magnitudes are not.
Across thirteen seed-saved runs (3 each on regex/json/math/sql, plus
the math seed=0 baseline), no two specialist shapes within a class
end on contradictory edits: regex always enables `testcases +
validate_retry`, math always enables `code_exec`, sql always enables
`results + validate_retry`.

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

### Split-sensitivity: do these gains survive different splits?

Five alternate random splits per class (split seeds 101-105), one
meta-seed per split, six iters each.  This is the binding-constraint
stress test: if the gains only show up on the hand-picked split, the
result is a split artefact.

| class | mean Δ ± SD | per-split Δ           | positive on |
|-------|------------:|-----------------------|-------------|
| regex | +12.4 ± 7.2 | +5, +5, +19, +19, +14 | 5/5         |
| json  |  +6.7 ± 8.7 | -5, +14, +14, +10, +0 | 3/5         |
| math  |  +7.5 ± 5.4 | +12, +0, +4, +8, +12  | 4/5         |

Regex generalises cleanly: every alternate split produces a positive
delta and the spread aligns with the per-seed spread on the
hand-picked split (+5 to +19 here vs +10 to +19 across seeds in the
main table).  Math is positive on 4/5 with one flat split, consistent
with the main-table per-seed spread of +5 to +17.  JSON is the
weakest: -5 on one split is the same overfit failure mode the per-seed
table shows.  The aggregate signal is robust on regex, present but
noisy on math, and right on the edge of statistical signal on json -
exactly the picture the per-seed numbers suggested, replicated under
a stronger test.

Saved in `runs/sensitivity/<class>/split{seed}_seed0.json`; aggregate
in `runs/sensitivity_summary.json`.

The trajectory plot below is what the loop actually does on the
hand-picked split: the dark-line drop happens between iter 0 (seed)
and iter 1 (the meta-agent's first edit), and most lineages plateau
within three accepted edits.

![evolution trajectories](runs/figures/trajectories.png)

### Cross-class transfer (test sets)

Each cell is the best-of-3-seeds specialist on the named class's test set.

| spec | regex | json | math | sql |
|---|---:|---:|---:|---:|
| seed              |  67% |  48% |  67% |  92% |
| regex specialist  | **86%** |  48% |  67% |  92% |
| json specialist   |  67% | **57%** |  67% |  92% |
| math specialist   |  67% |  38% | **83%** |  92% |
| sql specialist    |  71% |  43% |  62% | **100%** |

The diagonal is the best entry in every column.  Off-diagonals
mostly fall back toward seed; the math specialist drops 10 pp on
json (38% vs seed 48%) because its `code_exec`+python prompt is
irrelevant there and clutters the worker's input, and the sql
specialist drops 5 pp on math (62% vs seed 67%) for the same reason.
The two off-diagonals where a specialist beats seed by 4-5 pp (sql
spec on regex; regex spec on regex column - identical) are inside
the per-task noise floor and not intentional.  Sql is the only
column where everyone matches seed (92%): the seed already gets
near-saturation on sql, so the specialists' edits leave it alone or
just barely help.  The strong specialization signal is the diagonal
beating its column mean, plus the negative off-diagonals (irrelevant
edits actively hurting non-target classes) - both are visible.

![transfer matrix](runs/figures/transfer.png)

### Architecture ablations: what does the meta-agent actually need?

If the gains came from any-edit-to-the-spec rather than from
*structured* search guided by demos and failure feedback, the
architecture claim is overclaimed.  I ran three ablations against the
full meta-agent: random spec edit (no LLM proposer at all), no-demos
(LLM proposer with parent spec and failure feedback only, no demo
tasks), no-failure-feedback (LLM proposer with parent spec and demo
tasks only).  One meta-seed each, six iters, same hand-picked split.

| class | full Δ (n=3) | random Δ | no-demos Δ | no-feedback Δ |
|---|---:|---:|---:|---:|
| regex |  +16 |  +5 | +10  | +14 |
| json  |   +5 |   0 |  +5  |  +5 |
| math  |  +13 |   0 |   0  |  -8 |
| sql   |   +8 |   0 |  +8  |   0 |

Three things to take from this.  Random search is not the loop's
performance: every full-mode mean delta beats its random counterpart
on every class, and on math and sql the random baseline is exactly
zero.  Random can luck into the regex specialist (+5) because the
mutation space is small enough that picking `validate_retry` plus
`testcases` plus a non-zero `max_retries` happens roughly once in
twenty random draws and a single hit is enough to anchor the loop.
But on math the joint requirement (`code_exec`, prompt nudge, retry
budget) is sharp enough that random doesn't find it in six iters.

Demos help but aren't load-bearing: the no-demos column matches full
on json and sql, and only loses 6 pp on regex.  This is the easiest
to over-rotate on: the meta-agent's input includes the *current spec*
and the *current dev failures*, both of which carry class-shape signal
even without explicit demo tasks.  Demos help most on regex, where
the validator's per-task feedback is short and the meta-agent benefits
from seeing concrete (positives, negatives) pairs to reason about
backtick stripping or anchor placement.

Failure feedback is load-bearing for the tool-using classes.  Math
without failure feedback regresses below seed by 8 pp (the meta-agent
cannot tell which arithmetic items are failing, so its `code_exec`
proposal isn't anchored on real numbers, and on at least one iter it
proposes `validate_retry` instead, which fires no-op on math and burns
iters).  Sql without failure feedback fails to enable the validator
at all, so the +8 pp gap stays unrealised.  Regex still does fine
without feedback (+14) because the regex demos already include the
positive/negative test pattern.  The takeaway: failure feedback is
what turns the meta-agent from a "guess a reasonable shape" caller
into a "fix this specific failure" caller, and the gap between those
two is exactly the gain on math and sql.

Saved in `runs/ablation/<class>/<mode>_seed0.json`; aggregate in
`runs/ablation_summary.json`.

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
3. **Run multiple proposals per iter and pick the best by archive
   admission, instead of one proposal per iter.**  This changes the
   search from greedy to beam-2.  ADAS does something similar with an
   archive of candidates; my budget couldn't afford it but the
   architecture supports it.
4. **A fifth task class where neither `validate_retry` nor `code_exec`
   is the right answer.**  All four classes here resolve to one of
   those two patterns; I'd like to see how the loop behaves in a
   regime where the optimum is genuinely a system-prompt rewrite (the
   JSON case is the closest, and it's the noisiest result), so that
   the prompt-rewrite branch isn't a fallback when neither tool path
   helps.

## Appendix: four below-API observations from this experiment

The brief asked for "deep understanding of LLM mechanics below API level".
This is what the experiment actually let me observe, not generic LLM
trivia.

### A1. Formatting-prior leakage on raw fragments (regex backticks)

`gpt-4o-mini` wraps almost every regex output in single backticks, e.g.
` `^abc$` `.  The first version of my validator received the literal
backticks, refused to compile them, and the regex seed baseline sat at
17%.  Stripping a single-backtick wrap moved the baseline to ~71% on
the same prompts and same model with no other change.

I do not claim this is *caused by* BPE merges specifically.  Either
tokenization (single backticks are a low-cost token because they appear
in every code-fenced block in training data) or instruction-tuning
formatting priors (the model learned that "code" means "wrap in code
markers") could explain it.  What the experiment supports is the
weaker claim: when the model is asked for a fragment that its training
data almost always saw inside a markdown code-fence, its outputs leak
the framing tokens.  Anyone receiving raw fragments from a code model
should plan for this at the parser, not at the prompt; my fix is one
line of `_strip_one_backtick_fence` in `agent.py` and it pays for
itself by ~54 percentage points on the seed baseline.  For Mellum or
any code model emitting raw refactorings, the analogous fix lives in
the IDE-side receiver, not the prompt.

### A2. Tool-availability changes the action space; the prompt picks the branch

When the worker has `python_exec` available via OpenAI function-calling,
the assistant message is one of two structural choices: free-form
content, or a tool call.  Tool *availability* widens the action space
the model picks from on the next token.  It does not bias the model
toward picking tool-call.

The math experiment is the cleanest demonstration of that gap.  In an
early version, when the meta-agent proposed `tool_policy: code_exec`
without editing `system_prompt`, the worker *could* call `python_exec`
but mostly didn't.  Math specialist test score in that version was 50%,
*below* seed's 62%.  After I added one sentence to the meta-agent's
prompt instructing it to also edit the worker's system prompt
("tool-availability and tool-using-habit are coupled in the model's
behaviour but represented as independent fields in the spec"), the
specialist beat seed by 12-17 pp across two seeds.

The mechanism is not "tool schemas constrain every token to a different
distribution"; that's coarse.  The mechanism is that the model has a
strong prior toward direct-answer continuations on familiar question
shapes, and the tool branch of the action space is selected only when
the prompt makes the choice salient.  This is the same pattern that
shows up in production IDE agents: enabling a tool isn't enough; the
model needs to be told the tool is the *right* tool for the visible
task.

### A3. Decoding temperature serves different roles in a search loop

The worker uses `T=0.0`; the meta-agent uses `T=0.7`.  At `T=0.0` the
meta-agent was deterministic and the regression-rollback path was
never exercised in early runs (every proposal was monotone-or-better).
At `T=0.7` the proposal distribution widens enough that some
proposals regress.  Across the nine evolution runs in `runs/`
(3 classes x 3 seeds, 52 child proposals total), the lineage records:

- 20 archive additions (children that beat or matched parent on score)
- 9 dominated-but-accepted edits (same score, more tokens)
- 18 regression rollbacks (child strictly worse than parent on dev)
- 5 parse-apoptosis events (mostly the same `max_retries: 6` proposal
  repeated across consecutive iters by the same meta-agent instance,
  see A4 below)
- 0 smoke-apoptosis events (the worker never crashed on a smoke task)

The 18 regression rollbacks are the artefact of `T=0.7`: at `T=0.0`
they would not have happened, and the safeguard would have been
untested.

Two things to take from this.  First: in a search loop with a
gatekeeper, the right *role* of temperature isn't "diversity" in the
abstract - it's "exploration that the gatekeeper can afford to
reject".  Second: setting one temperature for the whole agent stack is
the easy default; it's also the wrong default when you have asymmetric
roles (deterministic worker + exploratory proposer).  This costs
nothing extra to change in the OpenAI SDK, and the literature on
agent-stack design rarely surfaces it explicitly.

### A4. The meta-agent has no memory across rejected iterations

This is the clearest below-API failure mode I observed, because it
shows up in the lineage logs.  The meta-agent's input on each iter is:
the system prompt, three demo tasks, the *current* parent spec, and
the *current* dev failures.  When a child is rejected (parse-apoptosis,
smoke-apoptosis, or regression rollback), the parent doesn't change,
so the dev failures don't change, so the next iter's input is
near-identical to the previous one.

The meta-agent is stateless across iters: it does not see its previous
proposal, the rejection cause, or its previous reasoning.  So when the
input is unchanged, it pattern-matches to the same theme again.

The math seed=0 lineage shows the cleanest case.  After iter 2, the
meta-agent proposed `max_retries: 6` (out of [0,4]) with the reason
"increasing max_retries allows more iterations".  That child was
rejected at parse-apoptosis.  Iter 3 input was identical to iter 2's,
because nothing had changed.  The meta-agent proposed `max_retries: 6`
*again*, with the reason "the worker is reaching the turn cap without
producing correct results, so increasing max_retries should help".
Same rejection.  Iter 4 input was *still* identical.  The meta-agent
proposed `max_retries: 6` for the third time, this time with the
reason "the worker reached the turn cap on two tasks, indicating it
needs more iterations".  The plateau detector finally stopped the
loop.

This is the mirror image of the failure mode the brief mentions in its
qualifications list ("context degradation, memory drift").  Here the
meta-agent has *too little* context-state, not too much: every iter
sees a fresh window.  The fix would be one of: (a) include the most
recent two rejected proposals and their rejection causes in the
meta-agent's input; (b) keep an episodic log of "things the meta-agent
has already tried with this parent" and surface it; (c) cool the
temperature each consecutive rejection so the proposer is forced to
deviate.  I have not implemented these; the observation alone was the
finding.

For an IDE agent that drives a multi-step refactor, the same pattern
shows up as "the agent keeps re-suggesting the same fix the user
already rejected".  The technical answer is the same: persist a
short-term episodic memory of rejected suggestions across turns.

## References

- ADAS - Hu, Lu, Clune.  *Automated Design of Agentic Systems*.  ICLR 2025.  arXiv:2408.08435.
- DGM - Sakana AI.  *Darwin Gödel Machine*.  2025.  arXiv:2505.22954.
- GEPA - Agrawal et al.  *Reflective prompt evolution*.  2025.  arXiv:2507.19457.
- Voyager - Wang et al.  *Skill library and curriculum in MineDojo*.  2023.  arXiv:2305.16291.
- Anthropic.  *Building effective agents*.  2024.  Canonical patterns: prompt chaining, routing, parallelisation, orchestrator-subagent, evaluator-optimizer.
