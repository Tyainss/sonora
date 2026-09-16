import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt
    import json

    import marimo as mo
    import polars as pl

    from sonora.breakout import (
        score_listening_share_momentum,
        score_recent_active_days,
        score_recent_listening_share,
    )
    from sonora.breakout.evaluation import (
        build_evaluation_periods,
        build_walk_forward_folds,
        evaluate_scores,
    )
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return (
        build_evaluation_periods,
        build_walk_forward_folds,
        dt,
        evaluate_scores,
        json,
        mo,
        paths,
        pl,
        score_listening_share_momentum,
        score_recent_active_days,
        score_recent_listening_share,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Behavioural baselines

    We test three simple behavioural questions:

    1. **Current importance:** are artists taking a larger share of the user's listening now more likely to break out?
    2. **Momentum:** are artists whose listening share is rising from one week to the next more likely to break out?
    3. **Repeated engagement:** are artists listened to across more distinct recent days more likely to break out?

    All use only listening available before the scoring date.
    """)
    return


@app.cell
def _(build_evaluation_periods, build_walk_forward_folds, dt, json, paths, pl):
    all_targets = pl.read_parquet(paths.daily_breakout_targets)
    all_breakout_events = pl.read_parquet(paths.breakout_events)
    listening_events = pl.read_parquet(paths.curated_listening_events)
    metadata = json.loads(
        paths.breakout_build_metadata.read_text(encoding="utf-8")
    )
    built_users = [user for user in metadata["users"] if user["status"] == "built"]
    if len(built_users) != 1:
        raise ValueError("This analysis notebook expects one built user")

    user_metadata = built_users[0]
    evaluation_user_id = user_metadata["user_id"]
    evaluation_start = dt.date(2022, 1, 1)
    observation_coverage_end = dt.date.fromisoformat(
        user_metadata["last_complete_date"]
    ) + dt.timedelta(days=1)
    periods = build_evaluation_periods(evaluation_start, observation_coverage_end)

    targets = all_targets.filter(pl.col("user_id") == evaluation_user_id)
    breakout_events = all_breakout_events.filter(
        pl.col("user_id") == evaluation_user_id
    )
    user_listening = listening_events.filter(
        pl.col("user_id") == evaluation_user_id
    )
    folds = build_walk_forward_folds(targets, periods)
    expected_candidates = pl.concat([fold.scoring for fold in folds])
    return (
        breakout_events,
        evaluation_start,
        expected_candidates,
        periods,
        user_listening,
    )


@app.cell
def _(
    breakout_events,
    evaluate_scores,
    evaluation_start,
    expected_candidates,
    periods,
    score_listening_share_momentum,
    score_recent_active_days,
    score_recent_listening_share,
    user_listening,
):
    share_scores = score_recent_listening_share(
        user_listening,
        expected_candidates,
    )
    momentum_scores = score_listening_share_momentum(
        user_listening,
        expected_candidates,
    )
    active_days_scores = score_recent_active_days(
        user_listening,
        expected_candidates,
    )
    share_evaluation = evaluate_scores(
        share_scores,
        expected_candidates,
        breakout_events,
        oos_start=evaluation_start,
        oos_end=periods[-1].scoring_end,
    )
    momentum_evaluation = evaluate_scores(
        momentum_scores,
        expected_candidates,
        breakout_events,
        oos_start=evaluation_start,
        oos_end=periods[-1].scoring_end,
    )
    active_days_evaluation = evaluate_scores(
        active_days_scores,
        expected_candidates,
        breakout_events,
        oos_start=evaluation_start,
        oos_end=periods[-1].scoring_end,
    )
    return (
        active_days_evaluation,
        active_days_scores,
        momentum_evaluation,
        momentum_scores,
        share_evaluation,
        share_scores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How do the three behaviours compare?

    Event coverage remains the first comparison. Timing then shows whether a different behavioural signal finds breakouts earlier rather than simply changing which events are detected.
    """)
    return


@app.cell
def _(active_days_evaluation, mo, momentum_evaluation, pl, share_evaluation):
    def _event_row(name, evaluation):
        summary = evaluation.event_summary.row(0, named=True)
        comparable = evaluation.event_results.filter(pl.col("is_comparable"))
        stages = (
            comparable.filter(pl.col("is_scoreable"))
            .group_by("timing_stage")
            .len()
        )
        stage_counts = {
            stage: stages.filter(pl.col("timing_stage") == stage)["len"].sum()
            for stage in ["Anticipatory", "Early", "Late", "Missed"]
        }
        return {
            "Baseline": name,
            "Detected / scoreable": (
                f"{summary['detected_events']} / {summary['scoreable_events']}"
            ),
            "Coverage@3": summary["event_coverage_at_3"],
            "Anticipatory": stage_counts["Anticipatory"],
            "Early": stage_counts["Early"],
            "Late": stage_counts["Late"],
            "Missed": stage_counts["Missed"],
            "Anticipatory rate": summary["anticipatory_detection_rate"],
            "Coverage through first period": summary[
                "coverage_through_first_period"
            ],
            "Median days to confirmation": summary[
                "days_to_confirmation_median"
            ],
            "Median days to breakout start": summary[
                "days_to_breakout_start_median"
            ],
        }

    event_comparison = pl.DataFrame(
        [
            _event_row("7-day listening share", share_evaluation),
            _event_row("7-day share momentum", momentum_evaluation),
            _event_row("7-day active days", active_days_evaluation),
        ]
    )
    mo.ui.table(event_comparison, selection=None)
    return


@app.cell(hide_code=True)
def _(active_days_evaluation, mo, momentum_evaluation, share_evaluation):
    _share = share_evaluation.event_summary.row(0, named=True)
    _momentum = momentum_evaluation.event_summary.row(0, named=True)
    _active = active_days_evaluation.event_summary.row(0, named=True)
    mo.md(f"""
    Current listening share detects **{_share['detected_events']} / {_share['scoreable_events']}** breakouts ({_share['event_coverage_at_3']:.1%}), momentum detects **{_momentum['detected_events']} / {_momentum['scoreable_events']}** ({_momentum['event_coverage_at_3']:.1%}), and repeated engagement detects **{_active['detected_events']} / {_active['scoreable_events']}** ({_active['event_coverage_at_3']:.1%}).

    Their anticipatory detections are **{_share['anticipatory_detections']}**, **{_momentum['anticipatory_detections']}**, and **{_active['anticipatory_detections']}**, respectively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Daily ranking quality

    This shows whether the Top 3 is more useful on individual days where at least one future breakout exists.
    """)
    return


@app.cell
def _(active_days_evaluation, mo, momentum_evaluation, pl, share_evaluation):
    def _daily_row(name, evaluation):
        daily = evaluation.daily_summary.row(0, named=True)
        return {
            "Baseline": name,
            "Scoring days": daily["scoring_days"],
            "Opportunity days": daily["opportunity_days"],
            "Mean Top-3 Capture": daily["mean_top3_capture"],
            "Mean NDCG@3": daily["mean_ndcg_at_3"],
        }

    daily_comparison = pl.DataFrame(
        [
            _daily_row("7-day listening share", share_evaluation),
            _daily_row("7-day share momentum", momentum_evaluation),
            _daily_row("7-day active days", active_days_evaluation),
        ]
    )
    mo.ui.table(daily_comparison, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stability over time

    The same comparison is shown by six-month confirmation period to make sure an overall result is not hiding one unusually strong or weak period.
    """)
    return


@app.cell
def _(
    active_days_evaluation,
    mo,
    momentum_evaluation,
    periods,
    pl,
    share_evaluation,
):
    period_status = pl.DataFrame(
        [
            {
                "confirmation_period": period.period,
                "Status": "Partial" if period.is_partial else "Complete",
            }
            for period in periods
        ]
    )
    share_periods = share_evaluation.period_summary.select(
        "confirmation_period",
        pl.col("event_coverage_at_3").alias("Share coverage@3"),
        pl.col("anticipatory_detections").alias("Share anticipatory"),
    )
    momentum_periods = momentum_evaluation.period_summary.select(
        "confirmation_period",
        pl.col("event_coverage_at_3").alias("Momentum coverage@3"),
        pl.col("anticipatory_detections").alias("Momentum anticipatory"),
    )
    active_days_periods = active_days_evaluation.period_summary.select(
        "confirmation_period",
        pl.col("event_coverage_at_3").alias("Active-days coverage@3"),
        pl.col("anticipatory_detections").alias("Active-days anticipatory"),
    )
    period_comparison = (
        period_status.join(
            share_periods,
            on="confirmation_period",
            how="left",
            validate="1:1",
        )
        .join(
            momentum_periods,
            on="confirmation_period",
            how="left",
            validate="1:1",
        )
        .join(
            active_days_periods,
            on="confirmation_period",
            how="left",
            validate="1:1",
        )
    )
    mo.ui.table(period_comparison, selection=None)
    return


@app.cell(hide_code=True)
def _(active_days_evaluation, mo, momentum_evaluation, share_evaluation):
    _share_event = share_evaluation.event_summary.row(0, named=True)
    _momentum_event = momentum_evaluation.event_summary.row(0, named=True)
    _active_event = active_days_evaluation.event_summary.row(0, named=True)
    _share_daily = share_evaluation.daily_summary.row(0, named=True)
    _momentum_daily = momentum_evaluation.daily_summary.row(0, named=True)
    _active_daily = active_days_evaluation.daily_summary.row(0, named=True)
    mo.md(f"""
    ## Conclusion

    - **Current importance:** {_share_event['event_coverage_at_3']:.1%} coverage, {_share_event['anticipatory_detections']} anticipatory detections, {_share_daily['mean_top3_capture']:.1%} mean Top-3 Capture.
    - **Momentum:** {_momentum_event['event_coverage_at_3']:.1%} coverage, {_momentum_event['anticipatory_detections']} anticipatory detections, {_momentum_daily['mean_top3_capture']:.1%} mean Top-3 Capture.
    - **Repeated engagement:** {_active_event['event_coverage_at_3']:.1%} coverage, {_active_event['anticipatory_detections']} anticipatory detections, {_active_daily['mean_top3_capture']:.1%} mean Top-3 Capture.

    **7-day listening share is the strongest simple behavioural reference.** It has the highest Event Coverage@3 and daily Top-3 Capture, while remaining competitive on timing.

    Momentum provides a small amount of earlier signal, but loses some coverage. Repeated engagement does not improve enough on either coverage or timing to replace current listening share.

    The main limitation is now clear: simple recent behaviour is very good at detecting breakouts once they are developing, but much weaker at anticipating them before breakout behaviour begins.

    Phase 3 should therefore focus on whether richer behavioural features can move useful detections earlier without sacrificing the strong coverage already achieved by this baseline.
    """)
    return


@app.cell(hide_code=True)
def _(
    active_days_evaluation,
    active_days_scores,
    expected_candidates,
    momentum_evaluation,
    momentum_scores,
    share_evaluation,
    share_scores,
):
    assert share_scores.height == expected_candidates.height
    assert momentum_scores.height == expected_candidates.height
    assert active_days_scores.height == expected_candidates.height
    assert share_evaluation.rankings.height == expected_candidates.height
    assert momentum_evaluation.rankings.height == expected_candidates.height
    assert active_days_evaluation.rankings.height == expected_candidates.height
    assert share_scores["score"].null_count() == 0
    assert momentum_scores["score"].null_count() == 0
    assert active_days_scores["score"].null_count() == 0
    return


if __name__ == "__main__":
    app.run()
