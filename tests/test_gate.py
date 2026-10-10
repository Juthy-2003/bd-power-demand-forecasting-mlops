from powerfc.registry.promote import decide

GOOD = {
    **{f"skill_smape_{b}h": 0.4 for b in (24, 48, 72)},
    **{f"coverage80_{b}h": 0.8 for b in (24, 48, 72)},
    **{f"smape_{b}h": s for b, s in ((24, 4.0), (48, 5.0), (72, 6.0))},
    "predict_latency_ms": 200.0,
}


def test_first_model_needs_only_the_absolute_guards():
    assert decide(GOOD, None, same_recipe=False).kind == "first"


def test_must_beat_seasonal_naive():
    bad = {**GOOD, "skill_smape_72h": -0.01}
    d = decide(bad, None, same_recipe=False)
    assert not d.promote and "72h" in d.reasons[0]


def test_coverage_band():
    assert decide({**GOOD, "coverage80_72h": 0.61}, None, False).promote  # what raw intervals gave in the backtest
    assert not decide({**GOOD, "coverage80_24h": 0.50}, None, False).promote  # collapsed intervals
    assert not decide({**GOOD, "coverage80_48h": 0.95}, None, False).promote  # uselessly wide
    assert decide({**GOOD, "coverage80_48h": 0.90}, None, False).promote


def test_new_recipe_must_beat_champion_by_margin_everywhere():
    champion = dict(GOOD)
    better_but_not_at_72 = {**GOOD, "smape_24h": 3.5, "smape_48h": 4.5, "smape_72h": 5.99}
    d = decide(better_but_not_at_72, champion, same_recipe=False)
    assert not d.promote and any("72h" in r for r in d.reasons)
    clearly_better = {**GOOD, "smape_24h": 3.5, "smape_48h": 4.5, "smape_72h": 5.5}
    assert decide(clearly_better, champion, same_recipe=False).kind == "improvement"


def test_refresh_allows_small_noise_but_not_regression():
    champion = dict(GOOD)
    assert decide({**GOOD, "smape_24h": 4.1}, champion, same_recipe=True).kind == "refresh"
    assert not decide({**GOOD, "smape_24h": 4.5}, champion, same_recipe=True).promote


def test_latency_budget():
    assert not decide({**GOOD, "predict_latency_ms": 10_000}, None, False).promote


def test_same_recipe_same_data_is_not_promoted():
    d = decide(GOOD, dict(GOOD), same_recipe=True, same_data=True)
    assert not d.promote and d.kind == "unchanged"
