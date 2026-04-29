"""Hand-tuned strong-spec baseline per class.

A reviewer can correctly ask "+13pp over a deliberately weak seed isn't
impressive if a 10-minute hand-written spec also clears that gap."  This
script writes one strong spec per class by hand and scores it on the test
set so the headline table can show meta-agent gains over a real baseline,
not just over the seed.

The hand specs are written from what I would have built knowing the class
in advance, before seeing any meta-agent output. They're not copied from
the specialist runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import SeedSpec
from eval import evaluate, load, split


HAND_SPECS: dict[str, dict] = {
    "regex": {
        # Same observation that got the seed from 17 to 71: strip backticks.
        # Beyond that, telling the model to anchor with ^...$ and to think
        # about the negatives is most of what a human would write.
        "system_prompt": (
            "You write Python regular expressions. Output ONLY the regex pattern, "
            "no quotes, no backticks, no explanation. Anchor with ^ and $. Use "
            "character classes that match the positives, then check that none of "
            "the negatives are accidentally accepted. If the task says 'phone' or "
            "'email' or 'IPv4', match the conventional shape exactly."
        ),
        "validation": "testcases",
        "tool_policy": "validate_retry",
        "max_retries": 2,
    },
    "json": {
        # Closed-schema JSON. Spell out the categories explicitly because the
        # value-mismatch failure mode is pure judgement, not parsing.
        "system_prompt": (
            "You output a JSON object with EXACTLY these keys: severity, kind, area. "
            "No prose, no fences. severity is one of [critical, high, med, low]: "
            "critical = production down or data loss; high = security flaw or major "
            "broken flow; med = a single feature broken with a workaround; low = "
            "polish, docs, or minor inconvenience. kind is one of [bug, feature, "
            "regression, support, docs]: regression only if the report says "
            "'used to work' or names a recent version. area is the affected component."
        ),
        "validation": "schema",
        "tool_policy": "validate_retry",
        "max_retries": 1,
    },
    "math": {
        # gpt-4o-mini compounds arithmetic errors. Tell it to use python_exec.
        "system_prompt": (
            "Solve the math word problem. For any computation with more than two "
            "operations, call python_exec with a short program that prints the "
            "final number. Otherwise answer directly. The final line of your "
            "answer must be the number alone, no units, no explanation. Round to "
            "two decimals if the answer is fractional."
        ),
        "validation": "none",
        "tool_policy": "code_exec",
        "max_retries": 2,
    },
    "sql": {
        # SQL against the small fixture. Spider/BIRD-style schema-injection
        # is already done in agent._user_msg; just need a strong prompt.
        "system_prompt": (
            "You output a single SQL query for SQLite. No prose, no fences, no "
            "trailing semicolon. Use exactly the table and column names from the "
            "schema in the user message. Prefer explicit JOIN ... ON over "
            "comma joins. When the question asks for a count, COUNT(*); when it "
            "asks for a top-N, ORDER BY ... DESC LIMIT N."
        ),
        "validation": "results",
        "tool_policy": "validate_retry",
        "max_retries": 1,
    },
}


def main() -> None:
    rows = []
    for cls in ["regex", "json", "math", "sql"]:
        spec = HAND_SPECS[cls]
        test = split(load(cls))["test"]
        seed_res = evaluate(SeedSpec, test, cls, label=f"{cls}/seed test")
        hand_res = evaluate(spec, test, cls, label=f"{cls}/hand strong test")
        rows.append({
            "class": cls,
            "n_test": hand_res.n_total,
            "seed": seed_res.score,
            "hand": hand_res.score,
            "delta_seed_to_hand": hand_res.score - seed_res.score,
            "hand_spec": spec,
        })
        print(f"{cls:5s}: seed {seed_res.score:.0%}  hand {hand_res.score:.0%}  "
              f"hand-vs-seed {hand_res.score - seed_res.score:+.0%}")

    Path("runs/hand_baseline.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote runs/hand_baseline.json")


if __name__ == "__main__":
    main()
