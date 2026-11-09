import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt
    import json

    import marimo as mo
    import polars as pl

    from sonora.breakout.evaluation import (
        build_evaluation_periods,
        build_walk_forward_folds,
        evaluate_scores,
        last_evaluable_scoring_date,
    )
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return (
        build_evaluation_periods,
        build_walk_forward_folds,
        dt,
        evaluate_scores,
        json,
        last_evaluable_scoring_date,
        mo,
        paths,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelling evaluation analysis

    Before comparing baselines or models, we need to make sure the historical comparison is fair: training must not use future outcomes too early, scoring must cover the expected dates, and breakouts should only count as misses when Sonora had a real chance to find them.
    """)
    return


@app.cell
def _(
    build_evaluation_periods,
    dt,
    json,
    last_evaluable_scoring_date,
    paths,
    pl,
):
    all_targets = pl.read_parquet(paths.daily_breakout_targets)
    all_breakout_events = pl.read_parquet(paths.breakout_events)
    artists = pl.read_parquet(paths.curated_artists)
    metadata = json.loads(
        paths.breakout_build_metadata.read_text(encoding="utf-8")
    )
    built_users = [user for user in metadata["users"] if user["status"] == "built"]
    if len(built_users) != 1:
        raise ValueError("This validation notebook expects one built user")

    user_metadata = built_users[0]
    evaluation_user_id = user_metadata["user_id"]
    targets = all_targets.filter(pl.col("user_id") == evaluation_user_id)
    breakout_events = all_breakout_events.filter(
        pl.col("user_id") == evaluation_user_id
    )

    first_scoreable_date = targets["scoring_date"].min()
    evaluation_start = dt.date(2022, 1, 1)
    observation_coverage_end = dt.date.fromisoformat(
        user_metadata["last_complete_date"]
    ) + dt.timedelta(days=1)
    last_evaluable_date = last_evaluable_scoring_date(observation_coverage_end)
    periods = build_evaluation_periods(evaluation_start, observation_coverage_end)
    return (
        artists,
        breakout_events,
        evaluation_start,
        evaluation_user_id,
        first_scoreable_date,
        last_evaluable_date,
        observation_coverage_end,
        periods,
        targets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What history can we evaluate?

    For this user, the current modelling cycle uses 2021 as the learning period and starts historical scoring on **1 January 2022**. Because every score needs 60 future days to know its outcome, evaluation stops 60 days before the end of complete listening coverage.
    """)
    return


@app.cell
def _(
    dt,
    evaluation_start,
    evaluation_user_id,
    first_scoreable_date,
    last_evaluable_date,
    mo,
    observation_coverage_end,
    periods,
    pl,
):
    boundaries = pl.DataFrame(
        [
            {
                "User": evaluation_user_id,
                "First scoreable day": first_scoreable_date,
                "Evaluation starts": evaluation_start,
                "Listening observed through": observation_coverage_end
                - dt.timedelta(days=1),
                "Last evaluable scoring day": last_evaluable_date,
            }
        ]
    )
    scoring_periods = pl.DataFrame(
        [
            {
                "Period": period.period,
                "Scoring starts": period.scoring_start,
                "Scoring ends": period.scoring_end,
                "Status": "Partial" if period.is_partial else "Complete",
            }
            for period in periods
        ]
    )
    mo.vstack(
        [
            mo.ui.table(boundaries, selection=None),
            mo.ui.table(scoring_periods, selection=None),
        ]
    )
    return boundaries, scoring_periods


@app.cell(hide_code=True)
def _(last_evaluable_date, mo, periods):
    _complete_periods = sum(not period.is_partial for period in periods)
    _last_status = "partial" if periods[-1].is_partial else "complete"
    mo.md(f"""
    This gives **{len(periods)} evaluation periods**. **{_complete_periods}** are complete and the last period is **{_last_status}**, ending on **{last_evaluable_date}**.

    The dates line up with the intended 60-day prediction window, so recent days whose outcomes are still unknown are not included in evaluation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Does each period train only on outcomes already known?

    At the start of each six-month period, training may use only rows whose full 60-day outcome window has already finished. The training history then grows as we move forward through time.
    """)
    return


@app.cell
def _(breakout_events, build_walk_forward_folds, mo, periods, pl, targets):
    folds = build_walk_forward_folds(targets, periods)
    fold_summary = pl.DataFrame(
        [
            {
                "Period": fold.period,
                "Training date": fold.training_cutoff,
                "Latest outcome date used": fold.training["horizon_end_date"].max(),
                "Training rows": fold.training_rows,
                "Scoring rows": fold.scoring_rows,
                "Breakouts confirmed": breakout_events.filter(
                    pl.col("confirmation_date").is_between(
                        fold.scoring_start, fold.scoring_end
                    )
                ).height,
                "Status": "Partial" if fold.is_partial else "Complete",
            }
            for fold in folds
        ]
    )
    mo.ui.table(fold_summary, selection=None)
    return fold_summary, folds


@app.cell(hide_code=True)
def _(fold_summary, mo, pl):
    _safe = fold_summary.select(
        (pl.col("Latest outcome date used") < pl.col("Training date")).all()
    ).item()
    _first_training_rows = fold_summary["Training rows"][0]
    _last_training_rows = fold_summary["Training rows"][-1]
    mo.md(f"""
    Every period passes the key check: the latest outcome used for training is **before** that period's training date.

    The available training data grows from **{_first_training_rows:,}** rows in the first period to **{_last_training_rows:,}** in the latest one. This is the expected behaviour for expanding training history.
    """)
    assert _safe
    return


@app.cell
def _(breakout_events, evaluate_scores, evaluation_start, folds, periods, pl):
    expected_candidates = pl.concat([fold.scoring for fold in folds])
    smoke_scores = expected_candidates.select(
        "user_id",
        "artist_id",
        "scoring_date",
    ).with_columns(
        (
            -pl.int_range(1, pl.len() + 1, dtype=pl.Int64)
            .over("user_id", "scoring_date")
            .cast(pl.Float64)
        ).alias("score")
    )
    evaluation = evaluate_scores(
        smoke_scores,
        expected_candidates,
        breakout_events,
        oos_start=evaluation_start,
        oos_end=periods[-1].scoring_end,
    )
    return evaluation, expected_candidates


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How many breakouts can we evaluate fairly?

    For a fair comparison, we only use breakouts where we have the full 60 days of historical rankings before confirmation. We also check whether the artist was actually eligible to be scored, and whether Sonora had a chance to surface them before the breakout started.
    """)
    return


@app.cell
def _(artists, evaluation, mo, pl):
    _events = evaluation.event_results
    _comparable = _events.filter(pl.col("is_comparable"))
    _comparable_scoreable = _comparable.filter(pl.col("is_scoreable"))
    _comparable_anticipatable = _comparable.filter(pl.col("is_anticipatable"))

    opportunity_summary = pl.DataFrame(
        [
            {
                "Measure": "Historical breakout events",
                "Events": _events.height,
            },
            {
                "Measure": "Full 60 days available for comparison",
                "Events": _comparable.height,
            },
            {
                "Measure": "At least one eligible scoring day",
                "Events": _comparable_scoreable.height,
            },
            {
                "Measure": "Could be scored before breakout started",
                "Events": _comparable_anticipatable.height,
            },
            {
                "Measure": "No chance to be scored in evaluation",
                "Events": _events.filter(~pl.col("is_scoreable")).height,
            },
        ]
    )

    _example_groups = [
        (
            "Could score before breakout",
            _events.filter(pl.col("is_comparable") & pl.col("is_anticipatable")),
        ),
        (
            "Could only score after breakout started",
            _events.filter(pl.col("is_comparable") & ~pl.col("is_anticipatable")),
        ),
        (
            "No scoring opportunity",
            _events.filter(~pl.col("is_scoreable")),
        ),
    ]
    _examples = []
    for _case, _group in _example_groups:
        if _group.height:
            _examples.append(
                _group.sort("confirmation_date")
                .head(1)
                .with_columns(pl.lit(_case).alias("Case"))
            )
    event_examples = (
        pl.concat(_examples)
        .join(
            artists.select("artist_id", "canonical_name"),
            on="artist_id",
            how="left",
            validate="m:1",
        )
        .select(
            "Case",
            pl.col("canonical_name").alias("Artist"),
            pl.col("breakout_period_start").alias("Breakout starts"),
            pl.col("confirmation_date").alias("Confirmed"),
            pl.col("valid_opportunities").alias("Eligible scoring days"),
            pl.col("pre_start_opportunities").alias("Before-start scoring days"),
        )
    )

    mo.vstack(
        [
            mo.ui.table(opportunity_summary, selection=None),
            mo.md("### Examples"),
            mo.ui.table(event_examples, selection=None),
        ]
    )
    return event_examples, opportunity_summary


@app.cell(hide_code=True)
def _(evaluation, mo, pl):
    _events = evaluation.event_results
    _comparable = _events.filter(pl.col("is_comparable"))
    _scoreable = _comparable.filter(pl.col("is_scoreable")).height
    _anticipatable = _comparable.filter(pl.col("is_anticipatable")).height
    _unscoreable = _events.filter(~pl.col("is_scoreable")).height
    mo.md(f"""
    **{_comparable.height} breakout events** have the full 60 days needed for a fair comparison, and **{_scoreable}** of them have at least one day where Sonora could score the artist.

    **{_anticipatable}** could be scored before the breakout started. The **{_unscoreable}** historical events where Sonora had no chance to score the artist stay visible, but they are not counted as misses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Does the evaluator run end to end?

    We use an artificial daily ordering only to make sure the evaluator works from start to finish. It is not a baseline, so its performance numbers are deliberately not shown.
    """)
    return


@app.cell
def _(evaluation, evaluation_start, expected_candidates, mo, periods, pl):
    _expected_days = expected_candidates.select("user_id", "scoring_date").n_unique()
    _unscoreable_are_separate = evaluation.event_results.filter(
        ~pl.col("is_scoreable") & (pl.col("timing_stage") != "Unscoreable")
    ).is_empty()
    checks = pl.DataFrame(
        [
            {
                "Check": "Every scoring candidate received one rank",
                "Result": "Passed"
                if evaluation.rankings.height == expected_candidates.height
                else "Failed",
            },
            {
                "Check": "Rankings stay inside the evaluation dates",
                "Result": "Passed"
                if evaluation.rankings["scoring_date"].min() >= evaluation_start
                and evaluation.rankings["scoring_date"].max()
                <= periods[-1].scoring_end
                else "Failed",
            },
            {
                "Check": "Daily metrics cover every scoring day",
                "Result": "Passed"
                if evaluation.daily_metrics.height == _expected_days
                else "Failed",
            },
            {
                "Check": "Events with no scoring opportunity stay separate",
                "Result": "Passed" if _unscoreable_are_separate else "Failed",
            },
        ]
    )
    mo.ui.table(checks, selection=None)
    return checks


@app.cell(hide_code=True)
def _(checks, mo, pl):
    _all_passed = checks.filter(pl.col("Result") != "Passed").is_empty()
    mo.md(
        "All checks pass. The evaluation framework can take a complete daily score stream, "
        "rank every eligible artist, and handle breakout opportunities as intended."
        if _all_passed
        else "At least one validation check failed and should be reviewed before using the evaluator."
    )
    assert _all_passed
    return


@app.cell(hide_code=True)
def _(evaluation, last_evaluable_date, mo, periods, pl):
    _comparable = evaluation.event_results.filter(pl.col("is_comparable"))
    _anticipatable = _comparable.filter(pl.col("is_anticipatable")).height
    mo.md(f"""
    ## 5. Conclusion

    The framework produces **{len(periods)} historical scoring periods** through **{last_evaluable_date}** without using future outcomes too early.

    The main comparison set contains **{_comparable.height} breakout events**. Of those, **{_anticipatable}** give Sonora at least one chance to surface the artist before breakout behaviour begins.

    The evaluation framework is ready to use for behavioural baselines. The ranking metrics will become meaningful once Phase 2 provides real baseline scores.
    """)
    return


@app.cell(hide_code=True)
def _(
    evaluation,
    evaluation_start,
    expected_candidates,
    first_scoreable_date,
    last_evaluable_date,
    periods,
):
    assert first_scoreable_date <= evaluation_start
    assert periods[0].scoring_start == evaluation_start
    assert expected_candidates["scoring_date"].max() == last_evaluable_date
    assert evaluation.rankings.height == expected_candidates.height
    return


if __name__ == "__main__":
    app.run()
