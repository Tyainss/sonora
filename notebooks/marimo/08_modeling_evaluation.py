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
    # Modelling evaluation framework

    This notebook validates the common evaluation protocol against the saved Sonora data. Reusable logic lives in `sonora.breakout.evaluation`.

    The smoke-test score below is synthetic and has **no modelling meaning**. It provides a complete daily ordering only to exercise the ranking, event, and daily-metric contracts.
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
        metadata,
        observation_coverage_end,
        periods,
        targets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Temporal protocol

    The evaluation start is a declared input for the user being evaluated. This modelling cycle starts on 2022-01-01. Six-month calendar folds continue from that date, with partial periods retained when needed.
    """)
    return


@app.cell
def _(
    evaluation_start,
    evaluation_user_id,
    first_scoreable_date,
    last_evaluable_date,
    mo,
    observation_coverage_end,
    periods,
    pl,
):
    temporal_boundaries = pl.DataFrame(
        [
            {
                "user_id": evaluation_user_id,
                "observation_coverage_end": observation_coverage_end,
                "first_scoreable_date": first_scoreable_date,
                "evaluation_start": evaluation_start,
                "last_evaluable_scoring_date": last_evaluable_date,
            }
        ]
    )
    period_table = pl.DataFrame(
        [
            {
                "period": period.period,
                "scoring_start": period.scoring_start,
                "scoring_end": period.scoring_end,
                "is_partial": period.is_partial,
            }
            for period in periods
        ]
    )
    mo.vstack(
        [
            mo.md("### Derived boundaries"),
            mo.ui.table(temporal_boundaries, selection=None),
            mo.md("### Calendar scoring periods"),
            mo.ui.table(period_table, selection=None),
        ]
    )
    return period_table, temporal_boundaries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fold availability

    `training_cutoff` is the model-fit date. The maximum training horizon must be strictly earlier than that date, including for positive rows whose confirmation was already known.
    """)
    return


@app.cell
def _(breakout_events, build_walk_forward_folds, periods, pl, targets):
    folds = build_walk_forward_folds(targets, periods)
    fold_table = pl.DataFrame(
        [
            {
                "period": fold.period,
                "training_cutoff": fold.training_cutoff,
                "maximum_training_horizon": fold.training[
                    "horizon_end_date"
                ].max(),
                "training_rows": fold.training_rows,
                "resolved_positive_training_rows": int(
                    fold.training["target_60d"].sum()
                ),
                "scoring_rows": fold.scoring_rows,
                "scoring_positive_rows": int(
                    fold.scoring["target_60d"].sum()
                ),
                "breakout_confirmations": breakout_events.filter(
                    pl.col("confirmation_date").is_between(
                        fold.scoring_start, fold.scoring_end
                    )
                ).height,
                "is_partial": fold.is_partial,
            }
            for fold in folds
        ]
    )
    fold_table
    return fold_table, folds


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
    return evaluation, expected_candidates, smoke_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Event opportunity analysis

    Main comparable events require their entire 60-day window inside OOS scoring. Other historical events remain visible, and zero-opportunity events remain unscoreable rather than becoming model misses.
    """)
    return


@app.cell
def _(artists, evaluation, mo, pl):
    event_opportunity_summary = evaluation.opportunity_summary
    opportunity_distribution = (
        evaluation.event_results.group_by("valid_opportunities")
        .len(name="events")
        .sort("valid_opportunities")
    )
    event_sample = (
        evaluation.event_results.join(
            artists.select("artist_id", "canonical_name"),
            on="artist_id",
            how="left",
            validate="m:1",
        )
        .select(
            "user_id",
            "canonical_name",
            "confirmation_date",
            "valid_opportunities",
            "pre_start_opportunities",
            "is_comparable",
            "timing_stage",
        )
        .head(12)
    )
    mo.vstack(
        [
            mo.ui.table(event_opportunity_summary, selection=None),
            mo.md("### Valid opportunities per event"),
            mo.ui.table(opportunity_distribution, selection=None),
            mo.md("### Event sample"),
            mo.ui.table(event_sample, selection=None),
        ]
    )
    return event_opportunity_summary, event_sample, opportunity_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluator smoke test

    These results validate plumbing only. The synthetic score is not a behavioural baseline and must not be interpreted as model performance.
    """)
    return


@app.cell
def _(evaluation, mo):
    mo.vstack(
        [
            mo.md("### Event-level result sample"),
            mo.ui.table(evaluation.event_results.head(12), selection=None),
            mo.md("### Timing stages"),
            mo.ui.table(
                evaluation.event_results.group_by("timing_stage")
                .len()
                .sort("timing_stage"),
                selection=None,
            ),
            mo.md("### Pooled event summary"),
            mo.ui.table(evaluation.event_summary, selection=None),
            mo.md("### Confirmation-period summaries"),
            mo.ui.table(evaluation.period_summary, selection=None),
            mo.md("### Daily ranking summary"),
            mo.ui.table(evaluation.daily_summary, selection=None),
        ]
    )
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
