# devlog

Rough notes from building stem-agent. Mostly bugs I hit and what they
taught me. Dated when the bug got fixed, not when it was introduced.

---

## 2026-04-12 - the validator dispatch bug, ~75 minutes lost

I added `validate_retry` and ran the loop on regex. Dev score didn't
move at all between iters: 4/6, 4/6, 4/6, 4/6.  The meta-agent kept
proposing `validation: testcases` and `tool_policy: validate_retry`
and *those proposals were getting accepted* by my safeguard rules.
But nothing was happening on dev.

Spent a while staring at meta-agent reasoning fields, thought I had a
prompt issue.  Tried bumping `max_retries`. Same.  Tried temperature
1.0 on the meta-agent. Same.

Then printed `out.steps` for one run on `iso-time`. Only one step in
the list: `[llm] initial`.  No `[validate]` step. No `[llm] retry`.
The validator wasn't being called at all.

```python
# what I had
VALIDATORS = {"regex": validate_regex, "json": validate_schema}

# what the run() loop did
if spec["validation"] not in VALIDATORS:
    return out
validator = VALIDATORS[spec["validation"]]
```

`spec["validation"]` is `"testcases"` or `"schema"`.  I keyed the dict
on the *task class name* by accident.  `"testcases" not in {"regex",
"json"}` is True, so the function returned without calling the
validator.  About an hour of confused logging before I saw it.

Fixed by pairing `(policy, class)` explicitly:

```python
_VALIDATOR = {
    ("testcases", "regex"): validate_regex,
    ("schema",    "json"):  validate_schema,
}
```

Lesson, mostly for me: when you have two concept axes (policy name vs
class name) and they share string-shape, the bug isn't going to look
like a bug. It looks like the meta-agent being slow.  Always log the
validator dispatch decision before assuming the model is the problem.

## 2026-04-12 - backticks, baseline jumped from 17% to 71%

Same evening. Smoke-tested the seed agent on the 15-task regex
corpus, expecting maybe 50-60% pass rate.  Got 17%.

Pulled up the actual outputs. The model returned things like:

```
`^(?!.*\t).+$`
```

with single backticks.  My validator did `re.compile("`^(?!.*\t).+$`")`
which doesn't fail to compile - it parses as a literal-backtick regex.
Then it tries to match positives, none start with a backtick, all
fail. Validator says "should match but didn't".

This was easy to see once I printed the raw model output, but I had
been staring at "17% baseline, that's just the model being bad at
regex" for longer than I'd like to admit.

Added six lines:

```python
def _strip_one_backtick_fence(s: str) -> str:
    if len(s) >= 2 and s.startswith("`") and s.endswith("`") and "\n" not in s:
        return s[1:-1]
    return s
```

Re-ran the same prompts against the same model. 71%.  A 54-pp gain
from a string strip.

The interesting question is *why* gpt-4o-mini wraps regex in
backticks.  I think it's that the training data has every regex
inside a markdown code-fence, so the model's prior on "regex output"
includes the framing tokens.  Asking for a fragment without a fence
is asking the model to fight its own formatting prior.  The
appendix in WRITEUP.md spells out what's defensible there and what
isn't.

## 2026-04-19 - JSON dev rebalance, fake 100% became honest 48%

Added the json class. Ran the loop. Dev climbed 0% → 50% → 67% → 67%
→ 83% → 100%. Test was 52%.

The specialist's system_prompt at iter 6:

> ... 'critical' severity is for system-breaking issues, 'regression'
> is for functionality that worked previously but is now broken ...

Cool. Read my dev split. It was: `crash-startup`, `auth-bypass`,
`support-q`, `how-to-config`, `lint-rule`, `data-loss`,
`csrf-bypass`, `memory-leak-worker`, `billing-double-charge`,
`leak-token-logs`, `regress-pasted-code`, `regress-search-empty`.

Five criticals, four regressions, three lows.  The dev set was
*mostly* "tag this critical-regression".  The meta-agent learned to
do exactly that.  It hit 100% on dev.  On test (which is more
balanced), the rule "everything important is critical" doesn't hold,
so test stayed flat.

Rebalanced dev: 4 critical, 2 high, 5 low, 1 med, mixed kinds.  Reran.
Now dev tops out around 75-83% and test hits 48% (vs seed 43%).
The +5pp delta is honest, smaller, and reflects a real (small) gain
from the prompt rules.

This was the most useful single bug for understanding the loop.
"100% dev with 52% test" looks like a win and is a failure.  The
fix isn't fancier search; it's a less-biased dev set.  Which means
the 12-task dev split is the binding constraint on how much the loop
can be trusted, not the meta-agent or the safeguards.

## 2026-04-21 - the frontier wasn't being mutated in place

Pareto frontier code, first version.  Ran regex.  Final report:
`frontier size: 1`.  But the lineage trace showed iter 1, iter 3,
iter 6 all marked `FRONTIER+`.  Three additions, one entry?

```python
def update_frontier(frontier, cand):
    if any(p.dominates(cand) for p in frontier):
        return False, []
    removed = [p for p in frontier if cand.dominates(p)]
    new = [p for p in frontier if not cand.dominates(p)]
    new.append(cand)
    return True, removed
```

I built `new` and threw it away.  Frontier never grew.  Should have
been `frontier[:] = [...]` to mutate in place, or returned the new
list and reassigned at the call site.  Five-minute fix once I saw it,
embarrassing for forty minutes before that.

Also a useful reminder that `_, _ = fn()` followed by reading state
is a recipe for silent failure when fn is supposed to be mutating.

## 2026-04-19 - strict-improvement was wrong, here's why

I had child accepted iff `child.score > parent.score`.  Felt rigorous.
It killed the regex specialist: every iter on the 12-task dev
yielded 4/6, 4/6, 4/6, then validate_retry came in at iter 1, also
4/6 on dev (it fixed a *test* item that wasn't in dev), got rejected
because no improvement, parent stayed seed, and the run produced
`final spec = seed, test 82% = 82%`.

Switched to monotone (`>=`).  Useful mutations that fix a test item
without moving dev get to stick around. Cost: wording-only edits
sometimes get accepted.  The Pareto check on tokens partially
catches that ("more tokens, same score" gets dominated), and the
plateau counter handles the rest.

Lesson: when your dev set is small enough that one task flipping is
within noise, strict improvement is too sharp.  Monotone with a
patience counter is the right rule for small dev sets.

## 2026-05-01 - typed JSON feedback, mean unchanged but variance widened

Going back to JSON because +5pp is anaemic and I knew it.  Decided
to tag the failure mode the meta-agent sees, instead of giving it
generic "wrong" strings and hoping it figures out which lever to
pull.

I added four tags:

```
[parse-fail]         output didn't parse as JSON
[missing-field]      JSON missing a required key
[value-not-allowed]  JSON value outside allowed set
[value-mismatch]     well-formed JSON but wrong category vs gold
```

The first three are fixable by `validate_retry`.  The fourth is not.
Now the meta-agent's heuristic is sharp: see lots of `[value-mismatch]`,
edit the system_prompt with rules, don't bother with retry.

Re-ran 3 seeds.  Mean delta unchanged (+5pp before, ~+5pp after).
*Variance* widened: +14, -5, +5 across seeds instead of +5, +5
before.  The meta-agent now commits more confidently to whichever
fix the typed tag suggests.  When it's right, it's righter.  When it
overfits dev, it overfits harder.

The right read: typed feedback gives the meta-agent a sharper
instrument; it doesn't give it more data.  With a 12-task dev split
the binding constraint is still data, not signal.  Bigger dev sets
would let typed feedback turn into actual reliable gains rather than
higher-variance gains.

## 2026-05-04 - sql class, two bugs in the fixture

Added the sql class.  Wrote a 9-user / 14-product / 25-order sqlite
fixture and 30 tasks; ran seed evaluation.  Got 100% on demo and
83% on dev.  Looked at the failures: two of them were tasks where
the gold result is an empty result set ("users with no orders",
"products never ordered").  My fixture had no users without orders
and every product had at least one order, so gold returned 0 rows
on both, predicted returned 0 rows because the predictions also
landed there, and the validator's row-count check still fired.
The actual bug was: gold had `WHERE NOT EXISTS (SELECT 1 FROM
orders WHERE user_id = users.id)` returning 0 rows because every
user had an order.  Adding a user without orders (Iris, country
DE) and two products without orders (Eraser, Sticky notes) made
the gold non-empty and the predicted vs gold comparison
meaningful.

Second bug: the seed agent got 0% on its first SQL test run.  I
was asking it for a SQL query to a database whose schema it had
never been told about.  Compared with regex/json/math, where the
task itself contains all the inputs, sql is the first class where
the *environment* (the schema) is part of the worker's effective
prompt.  Spider/BIRD's convention is to prepend the schema to the
user message, which keeps the seed class-agnostic while letting
each class inject what it needs.  Added one helper (`_user_msg`
in `agent.py`) and a `SQL_SCHEMA_PREFIX` constant; seed jumped
from 0% to 92% on test.

The class-agnostic seed stays clean (one system prompt, no
class-specific knowledge), and the class-specific knowledge lives
in the worker's `_user_msg` helper.  If I added a fifth class with
its own context (a code base, a tool spec, a calendar), the same
pattern would extend.

---

Things I haven't fixed and would, with another week:

- The math eval normalizer is fragile. "Final answer rounds to 80.0"
  picks up "0" as the last number (after rstrip-zero) and fails. Need
  unit tests for the normalizer specifically, with messy outputs.
- `_score_json_full` re-parses the output after `validate_schema`
  already did. Two `json.loads` calls per task. Cheap but ugly.
- The runs/ directory has lineage.json files that include the full
  spec history. They get big quickly. Should probably split into
  meta-only and full-history versions for easier loading.
