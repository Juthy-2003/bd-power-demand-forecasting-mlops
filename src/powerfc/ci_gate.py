"""Promotion gate for CI: does this commit's model recipe beat the committed
champion recipe on the pinned data and a fixed backtest window?

The data is the checksum-pinned PGCB file up to the replay start
(`store.AS_OF_START`), so every CI run evaluates on identical folds. The
champion's metrics live in `ci/champion_metrics.json`; a pull request that
changes the recipe must beat them under the same rules as the live gate
(`registry.promote.decide`), or the job fails.

    python -m powerfc.ci_gate                 # compare, exit 1 if rejected
    python -m powerfc.ci_gate --write         # record this recipe as the new baseline
"""

import argparse
import json
import sys
from pathlib import Path

from powerfc import config
from powerfc.data.clean import clean
from powerfc.data.contract import validate_clean, validate_raw
from powerfc.data.load import fetch_raw, read_raw
from powerfc.data.store import AS_OF_START
from powerfc.registry.promote import decide
from powerfc.train import evaluate_recipe, gate_window, make_recipe, recipe_hash

BASELINE = Path("ci/champion_metrics.json")
WEEKS, REFIT_EVERY = 12, 4


def evaluate() -> dict:
    fetch_raw()
    raw = read_raw()
    data = validate_clean(clean(validate_raw(raw[raw["datetime"] <= AS_OF_START].reset_index(drop=True))))
    recipe = make_recipe()
    window = gate_window(data, WEEKS, REFIT_EVERY)
    _, metrics = evaluate_recipe(data, recipe, window)
    metrics["predict_latency_ms"] = 0.0  # latency is gated in the live pipeline, not on CI runners
    return {"recipe_hash": recipe_hash(recipe), "window": f"{window.start}..{window.end}", "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    result = evaluate()
    if a.write or not BASELINE.exists():
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"baseline written: recipe {result['recipe_hash']}")
        return
    base = json.loads(BASELINE.read_text())
    same = base["recipe_hash"] == result["recipe_hash"]
    decision = decide(result["metrics"], base["metrics"], same_recipe=same, same_data=same)
    table = {
        f"{b}h": {
            "champion": round(base["metrics"][f"smape_{b}h"], 3),
            "candidate": round(result["metrics"][f"smape_{b}h"], 3),
        }
        for b in config.BLOCKS
    }
    print(
        json.dumps(
            {
                "candidate": result["recipe_hash"],
                "champion": base["recipe_hash"],
                "smape": table,
                "decision": decision.kind,
                "reasons": decision.reasons,
            },
            indent=2,
        )
    )
    if same:
        print("recipe unchanged: nothing to gate")
        return
    sys.exit(0 if decision.promote else 1)


if __name__ == "__main__":
    main()
