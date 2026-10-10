"""The promotion gate — the only code that moves the `champion` alias.

Used by the Airflow DAG and by CI. Decision rules (answers to v01 §18 Q7–Q8,
decided 2026-10-10):

Always required (absolute guards):
  * beats seasonal-naive at every horizon block (sMAPE skill > 0);
  * raw P10–P90 coverage inside COVERAGE_BAND = [0.55, 0.92] at every block.
    The intervals are not calibrated, and in the 2-year backtest
    (2023-03 → 2025-03, measured 2026-10-10) they covered 68.3% / 65.6% /
    61.2% at 1–24 / 25–48 / 49–72 h instead of the nominal 80%. The lower
    edge sits about 6 points below the worst block, so it stops a model
    whose intervals have collapsed, not one that behaves like this recipe.
    The upper edge stops intervals so wide they are useless. The band was
    set from the backtest only, before the holdout was scored (ADR 0009).
  * predict latency for a 72-hour forecast within LATENCY_BUDGET_MS.

Then, against the current champion, evaluated on the SAME data version and
the SAME backtest folds:
  * different recipe → challenger sMAPE must be at least MARGIN (1%) lower
    at every block;
  * same recipe retrained on newer data ("refresh") → may be at most
    REFRESH_TOLERANCE (5%) worse than the champion's recorded sMAPE at any
    block. A refresh keeps the model current; it is not an improvement claim.
  * same recipe AND same data version → nothing is new; keep the champion.

A refused challenger is evidence too: the decision and its reasons are
written to the model version's tags.
"""

import json
from dataclasses import dataclass, field

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from powerfc import config

MARGIN = 0.01
REFRESH_TOLERANCE = 0.05
COVERAGE_BAND = (0.55, 0.92)
LATENCY_BUDGET_MS = 3000.0


@dataclass
class Decision:
    promote: bool
    kind: str  # first | improvement | refresh | unchanged | rejected
    reasons: list[str] = field(default_factory=list)


def decide(challenger: dict, champion: dict | None, same_recipe: bool, same_data: bool = False) -> Decision:
    if champion is not None and same_recipe and same_data:
        return Decision(False, "unchanged", ["same recipe and same data version as the champion; nothing to promote"])
    reasons: list[str] = []
    ok = True
    for b in config.BLOCKS:
        skill = challenger.get(f"skill_smape_{b}h", float("nan"))
        if not skill > 0:
            ok = False
            reasons.append(f"{b}h: does not beat seasonal-naive (skill {skill:.3f})")
        cov = challenger.get(f"coverage80_{b}h", float("nan"))
        if not COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1]:
            ok = False
            reasons.append(f"{b}h: 80% interval coverage {cov:.3f} outside {COVERAGE_BAND}")
    latency = challenger.get("predict_latency_ms", float("inf"))
    if latency > LATENCY_BUDGET_MS:
        ok = False
        reasons.append(f"predict latency {latency:.0f} ms > {LATENCY_BUDGET_MS:.0f} ms")
    if not ok:
        return Decision(False, "rejected", reasons)
    if champion is None:
        return Decision(True, "first", ["no champion yet; absolute guards passed"])

    for b in config.BLOCKS:
        new, old = challenger[f"smape_{b}h"], champion[f"smape_{b}h"]
        if same_recipe:
            if new > old * (1 + REFRESH_TOLERANCE):
                ok = False
                reasons.append(f"{b}h: refresh regressed sMAPE {old:.3f} → {new:.3f} (> {REFRESH_TOLERANCE:.0%})")
        elif new > old * (1 - MARGIN):
            ok = False
            reasons.append(f"{b}h: sMAPE {new:.3f} not {MARGIN:.0%} better than champion {old:.3f}")
    if not ok:
        return Decision(False, "rejected", reasons)
    kind = "refresh" if same_recipe else "improvement"
    detail = " ".join(f"{b}h {champion[f'smape_{b}h']:.3f}→{challenger[f'smape_{b}h']:.3f}" for b in config.BLOCKS)
    return Decision(True, kind, [f"{kind}: sMAPE {detail}"])


def _version_by_alias(client: MlflowClient, alias: str):
    try:
        return client.get_model_version_by_alias(config.REGISTERED_MODEL, alias)
    except MlflowException:
        return None


def _metrics(client: MlflowClient, run_id: str) -> dict:
    return dict(client.get_run(run_id).data.metrics)


def champion_metrics_on(client: MlflowClient, recipe_hash: str, data_version: str, window: str) -> dict | None:
    """Find an existing evaluation of `recipe_hash` on this data version and window."""
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=(
            f"tags.recipe_hash = '{recipe_hash}' and tags.data_version = '{data_version}' and tags.window = '{window}'"
        ),
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return dict(runs[0].data.metrics) if runs else None


def promote(evaluate_champion=None) -> Decision:
    """Compare the `challenger` with the `champion` and move the alias if it wins.

    `evaluate_champion(recipe, data_version, window) -> metrics` is called
    when the champion's recipe has no evaluation on the challenger's data
    version yet (e.g. after new data arrived with a different recipe)."""
    client = MlflowClient()
    challenger = _version_by_alias(client, "challenger")
    if challenger is None:
        raise RuntimeError("no model version has the 'challenger' alias")
    champ = _version_by_alias(client, "champion")
    ch_run = client.get_run(challenger.run_id)
    ch_metrics = dict(ch_run.data.metrics)
    ch_tags = ch_run.data.tags

    if champ is not None and champ.version == challenger.version:
        return Decision(False, "rejected", ["challenger is already the champion"])

    champion_metrics, same_recipe, same_data = None, False, False
    if champ is not None:
        champ_tags = client.get_run(champ.run_id).data.tags
        same_recipe = champ_tags["recipe_hash"] == ch_tags["recipe_hash"]
        same_data = champ_tags["data_version"] == ch_tags["data_version"]
        if same_recipe:
            champion_metrics = _metrics(client, champ.run_id)
        else:
            champion_metrics = champion_metrics_on(
                client, champ_tags["recipe_hash"], ch_tags["data_version"], ch_tags["window"]
            )
            if champion_metrics is None:
                if evaluate_champion is None:
                    raise RuntimeError("champion recipe has no evaluation on this data version")
                recipe = json.loads(client.get_run(champ.run_id).data.params.get("recipe_json", "{}") or "{}")
                champion_metrics = evaluate_champion(recipe, ch_tags["data_version"], ch_tags["window"])

    decision = decide(ch_metrics, champion_metrics, same_recipe, same_data)
    client.set_model_version_tag(config.REGISTERED_MODEL, challenger.version, "gate_decision", decision.kind)
    client.set_model_version_tag(
        config.REGISTERED_MODEL, challenger.version, "gate_reasons", "; ".join(decision.reasons)[:4900]
    )
    if decision.promote:
        client.set_registered_model_alias(config.REGISTERED_MODEL, "champion", challenger.version)
        if champ is not None:
            client.set_registered_model_alias(config.REGISTERED_MODEL, "previous-champion", champ.version)
    return decision
