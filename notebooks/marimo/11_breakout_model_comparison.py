import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from sonora.breakout import build_modeling_data, score_recent_listening_share
    from sonora.breakout.diagnostics import diagnose_lost_detections
    from sonora.breakout.evaluation import (
        EvaluationPeriod,
        build_evaluation_periods,
        build_walk_forward_folds,
        combine_score_streams,
        evaluate_scores,
    )
    from sonora.breakout.evaluation.comparison import compare_event_detections
    from sonora.breakout.evaluation.holdout import load_or_create_holdout
    from sonora.breakout.modeling import run_logistic_walk_forward
    from sonora.breakout.tracking import capture_comparison_context, record_comparison
    from sonora.data.paths import DEFAULT_DATA_PATHS

    paths = DEFAULT_DATA_PATHS
    return (
        EvaluationPeriod,
        Path,
        build_evaluation_periods,
        build_modeling_data,
        build_walk_forward_folds,
        capture_comparison_context,
        combine_score_streams,
        compare_event_detections,
        diagnose_lost_detections,
        dt,
        evaluate_scores,
        json,
        load_or_create_holdout,
        mo,
        paths,
        pl,
        record_comparison,
        run_logistic_walk_forward,
        score_recent_listening_share,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Can listening history help detect breakouts earlier?

    I compare **7-day listening share** with **Logistic Regression**, which combines that share with the number of earlier listens. Earlier listens exclude the most recent seven days and are log-transformed.

    Both rank the same artists on the same dates. Detection requires two Top-3 appearances within the 60 days before confirmation, reduced to one when only one eligible day exists.
    """)
    return


@app.cell
def _(
    EvaluationPeriod,
    Path,
    build_evaluation_periods,
    capture_comparison_context,
    dt,
    json,
    load_or_create_holdout,
    paths,
    pl,
):
    _metadata = json.loads(paths.breakout_build_metadata.read_text(encoding="utf-8"))
    _users = [user for user in _metadata["users"] if user["status"] == "built"]
    if len(_users) != 1:
        raise ValueError("This analysis notebook expects one built user")
    _user = _users[0]
    _user_id = _user["user_id"]
    _coverage_end = dt.date.fromisoformat(_user["last_complete_date"]) + dt.timedelta(days=1)
    holdout = load_or_create_holdout(
        Path("artifacts/breakout/first_ml_holdout.json"),
        user_id=_user_id,
        generation="first_ml",
        observation_coverage_end=_coverage_end,
    )
    comparison_context = capture_comparison_context(
        project_root=Path("."),
        paths=paths,
        build_metadata=_metadata,
        holdout_path=Path("artifacts/breakout/first_ml_holdout.json"),
    )
    evaluation_start = dt.date(dt.date.fromisoformat(_user["score_start"]).year + 1, 1, 1)
    periods = [
        EvaluationPeriod(
            period.period,
            period.scoring_start,
            min(period.scoring_end, holdout.development_end),
            period.is_partial or period.scoring_end > holdout.development_end,
        )
        for period in build_evaluation_periods(evaluation_start, _coverage_end)
        if period.scoring_start <= holdout.development_end
    ]
    if not periods:
        raise ValueError("No development scoring period precedes the holdout")
    development_targets = (
        pl.scan_parquet(paths.daily_breakout_targets)
        .filter((pl.col("user_id") == _user_id) & (pl.col("scoring_date") <= holdout.development_end))
        .collect()
    )
    development_events = (
        pl.scan_parquet(paths.breakout_events)
        .filter(
            (pl.col("user_id") == _user_id)
            & (pl.col("confirmation_date") <= holdout.development_end + dt.timedelta(days=1))
        )
        .collect()
    )
    user_listening = (
        pl.scan_parquet(paths.curated_listening_events)
        .filter(
            (pl.col("user_id") == _user_id)
            & (pl.col("listened_at") < dt.datetime.combine(holdout.lead_in_start, dt.time(), tzinfo=dt.UTC))
        )
        .collect()
    )
    artist_names = pl.read_parquet(paths.curated_artists).select("artist_id", "canonical_name")
    return (
        artist_names,
        comparison_context,
        development_events,
        development_targets,
        evaluation_start,
        holdout,
        periods,
        user_listening,
    )


@app.cell(hide_code=True)
def _(evaluation_start, holdout, mo):
    mo.md(f"""
    These rankings cover **{evaluation_start} to {holdout.development_end}**. The reserved prediction lead-in starts on **{holdout.lead_in_start}**, and the final event period starts on **{holdout.event_start}**.

    Each model trains on labels whose full 60-day outcome was already available at the start of its scoring period. Scaling is fitted on that training history.
    """)
    return


@app.cell
def _(
    build_modeling_data,
    build_walk_forward_folds,
    combine_score_streams,
    compare_event_detections,
    development_events,
    development_targets,
    evaluate_scores,
    evaluation_start,
    periods,
    pl,
    run_logistic_walk_forward,
    score_recent_listening_share,
    user_listening,
):
    _data = build_modeling_data(user_listening, development_targets)
    _folds = build_walk_forward_folds(_data, periods)
    expected_candidates = pl.concat([fold.scoring for fold in _folds])
    baseline_scores = score_recent_listening_share(user_listening, expected_candidates)
    fold_results = run_logistic_walk_forward(_data, periods, baseline_scores)
    _model_scores = combine_score_streams([result.scores for result in fold_results])
    baseline_evaluation = evaluate_scores(
        baseline_scores,
        expected_candidates,
        development_events,
        oos_start=evaluation_start,
        oos_end=periods[-1].scoring_end,
    )
    model_evaluation = evaluate_scores(
        _model_scores,
        expected_candidates,
        development_events,
        oos_start=evaluation_start,
        oos_end=periods[-1].scoring_end,
    )
    paired_events = compare_event_detections(
        baseline_evaluation.event_results, model_evaluation.event_results
    )
    return baseline_evaluation, fold_results, model_evaluation, paired_events


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Does the model preserve coverage and detect more artists early?

    Anticipatory means before breakout listening starts. Early means during its first important week. Late means after that week, but before confirmation.
    """)
    return


@app.cell
def _(baseline_evaluation, mo, model_evaluation, pl):
    def _row(name, evaluation):
        _summary = evaluation.event_summary.row(0, named=True)
        _events = evaluation.event_results.filter(pl.col("is_comparable") & pl.col("is_scoreable"))
        return {
            "Approach": name,
            "Detected / scoreable": f"{_summary['detected_events']} / {_summary['scoreable_events']}",
            "Coverage": None if _summary["event_coverage_at_3"] is None else f"{_summary['event_coverage_at_3']:.1%}",
            "Anticipatory": _summary["anticipatory_detections"],
            "Early": _events.filter(pl.col("timing_stage") == "Early").height,
            "Late": _events.filter(pl.col("timing_stage") == "Late").height,
            "Missed": _events.filter(pl.col("timing_stage") == "Missed").height,
            "Median days before start": _summary["days_to_breakout_start_median"],
        }
    mo.ui.table(
        pl.DataFrame([_row("7-day listening share", baseline_evaluation), _row("Logistic Regression", model_evaluation)]),
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(baseline_evaluation, mo, model_evaluation):
    _baseline = baseline_evaluation.event_summary.row(0, named=True)
    _model = model_evaluation.event_summary.row(0, named=True)
    mo.md(f"""
    The baseline detects **{_baseline['detected_events']}** breakouts and the model detects **{_model['detected_events']}**, from the same **{_baseline['scoreable_events']}** scoreable events. **{_baseline['anticipatable_events']}** had an eligible day before breakout listening started.

    The median timing uses detected events only. Positive days mean detection before breakout starts; negative days mean detection after it starts. The paired comparison below shows timing changes for the same events.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which detections moved earlier or later?
    """)
    return


@app.cell
def _(mo, paired_events, pl):
    _counts = dict(paired_events.group_by("change").len().iter_rows())
    mo.ui.table(
        pl.DataFrame(
            {
                "Change": ["Earlier", "Later", "Same day", "Newly detected", "Lost detection", "Missed by both"],
                "Events": [_counts.get(change, 0) for change in ["Earlier", "Later", "Same day", "Newly detected", "Lost detection", "Missed by both"]],
            }
        ),
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo, paired_events):
    _days = paired_events["days_earlier"].drop_nulls()
    _median = _days.median()
    _timing = "unavailable" if _median is None else f"{_median:g} days"
    mo.md(f"""
    For events both approaches detect, the median change is **{_timing}**. Positive values mean the model detects earlier; negative values mean it detects later.
    """)
    return


@app.cell
def _(artist_names, mo, paired_events, pl):
    _details = pl.concat(
        [
            paired_events.filter(pl.col("change").is_in(["Newly detected", "Lost detection"])),
            paired_events.filter(pl.col("change") == "Earlier").sort("days_earlier", descending=True).head(5),
            paired_events.filter(pl.col("change") == "Later").sort("days_earlier").head(5),
        ]
    ).join(artist_names, on="artist_id", how="left", validate="m:1").select(
        pl.col("canonical_name").fill_null(pl.col("artist_id")).alias("Artist"),
        pl.col("change").alias("Change"),
        pl.col("reference_detection_date").alias("Baseline date"),
        pl.col("model_detection_date").alias("Model date"),
        pl.col("days_earlier").alias("Days earlier"),
    )
    mo.accordion({"New/lost detections and the five largest changes in each direction": mo.ui.table(_details, selection=None)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Why did the model miss breakouts the baseline caught?

    Each event needs enough eligible Top-3 days before confirmation. The summary counts the full evaluation window; the details show days when either approach ranked the artist in its Top 3.
    """)
    return


@app.cell
def _(baseline_evaluation, diagnose_lost_detections, fold_results, model_evaluation):
    lost_diagnostics = diagnose_lost_detections(baseline_evaluation, model_evaluation, fold_results)
    return (lost_diagnostics,)


@app.cell
def _(artist_names, lost_diagnostics, mo, pl):
    _summary = lost_diagnostics.summary.join(artist_names, on="artist_id", how="left", validate="m:1").select(
        pl.col("canonical_name").fill_null(pl.col("artist_id")).alias("Artist"),
        pl.col("reference_detection_date").alias("Baseline detection"),
        pl.col("valid_opportunities").alias("Eligible days"),
        pl.col("required_top3_appearances").alias("Required Top-3 days"),
        pl.col("baseline_top3_days").alias("Baseline Top-3 days"),
        pl.col("model_top3_days").alias("Model Top-3 days"),
        pl.col("best_model_rank").alias("Best model rank"),
    )
    mo.ui.table(_summary, selection=None) if not _summary.is_empty() else mo.md("The model lost no detections in this comparison.")
    return


@app.cell
def _(artist_names, lost_diagnostics, mo, pl):
    mo.stop(lost_diagnostics.summary.is_empty())
    _cases = {}
    for _event in lost_diagnostics.summary.join(artist_names, on="artist_id", how="left", validate="m:1").to_dicts():
        _artist = _event["artist_id"]
        _user = _event["user_id"]
        _name = _event["canonical_name"] or _artist
        _days = lost_diagnostics.daily_ranks.filter(
            (pl.col("user_id") == _user) & (pl.col("artist_id") == _artist)
            & ((pl.col("baseline_rank") <= 3) | (pl.col("model_rank") <= 3))
        ).select(
            pl.col("scoring_date").alias("Date"),
            pl.col("baseline_rank").alias("Baseline rank"),
            pl.col("model_rank").alias("Model rank"),
            (100 * pl.col("listening_share_7d")).round(2).alias("Recent share (%)"),
            pl.col("previous_listening_count").cast(pl.Int64).alias("Earlier listens"),
        )
        _promotions = lost_diagnostics.displacements.filter(
            (pl.col("user_id") == _user) & (pl.col("artist_id") == _artist)
        ).join(
            artist_names.rename({"artist_id": "promoted_artist_id"}),
            on="promoted_artist_id", how="left", validate="m:1",
        ).select(
            pl.col("scoring_date").alias("Date"),
            pl.col("canonical_name").fill_null(pl.col("promoted_artist_id")).alias("Promoted artist"),
            pl.col("promoted_baseline_rank").alias("Baseline rank"),
            pl.col("promoted_model_rank").alias("Model rank"),
            (100 * pl.col("promoted_listening_share_7d")).round(2).alias("Recent share (%)"),
            pl.col("promoted_previous_listening_count").cast(pl.Int64).alias("Earlier listens"),
            pl.col("share_advantage").round(3).alias("Recent-share advantage"),
            pl.col("previous_count_advantage").round(3).alias("Earlier-listening advantage"),
            pl.col("model_advantage").round(3).alias("Model score lead"),
        )
        _cases[_name] = mo.vstack([
            mo.md("**Days in either Top 3**"),
            mo.ui.table(_days, selection=None, page_size=max(1, _days.height)),
            mo.md("**Artists promoted from outside the baseline Top 3**"),
            mo.ui.table(_promotions, selection=None, page_size=max(1, _promotions.height)),
        ])
    mo.accordion(_cases)
    return


@app.cell(hide_code=True)
def _(lost_diagnostics, mo, pl):
    mo.stop(lost_diagnostics.summary.is_empty())
    _promotions = lost_diagnostics.displacements
    _overturns = _promotions.filter(
        (pl.col("share_advantage") < 0)
        & (pl.col("previous_count_advantage") > 0)
        & (pl.col("model_advantage") > 0)
    ).height
    mo.md(f"""
    In **{_overturns} of {_promotions.height}** listed promotions, an earlier-listening advantage outweighed a recent-share disadvantage.

    The two feature advantages add up to the promoted artist's model score lead, using the fitted model for that day. These are decision-score differences, not probabilities. Several artists can enter or leave the Top 3 on the same day.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Is the result consistent across periods?

    Events are grouped by when their breakout was confirmed. Small event counts make percentages less reliable.
    """)
    return


@app.cell
def _(baseline_evaluation, mo, model_evaluation, periods, pl):
    _baseline = baseline_evaluation.period_summary.select(
        pl.col("confirmation_period").alias("Period"),
        pl.col("scoreable_events").alias("Events"),
        (100 * pl.col("event_coverage_at_3")).round(1).alias("Baseline coverage (%)"),
        pl.col("anticipatory_detections").alias("Baseline anticipatory"),
    )
    _model = model_evaluation.period_summary.select(
        pl.col("confirmation_period").alias("Period"),
        (100 * pl.col("event_coverage_at_3")).round(1).alias("Model coverage (%)"),
        pl.col("anticipatory_detections").alias("Model anticipatory"),
    )
    _status = pl.DataFrame({"Period": [period.period for period in periods], "Partial period": [period.is_partial for period in periods]})
    mo.ui.table(_baseline.join(_model, on="Period", validate="1:1").join(_status, on="Period", validate="1:1"), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Is the daily Top 3 more useful?

    Top-3 Capture measures how many future breakout artists fit into the available shortlist. NDCG gives more credit when they appear near the top. Both use days with at least one future breakout.
    """)
    return


@app.cell
def _(baseline_evaluation, mo, model_evaluation, pl):
    _rows = []
    for _name, _evaluation in [("7-day listening share", baseline_evaluation), ("Logistic Regression", model_evaluation)]:
        _summary = _evaluation.daily_summary.row(0, named=True)
        _rows.append({
            "Approach": _name,
            "Days with a future breakout": _summary["opportunity_days"],
            "Top-3 Capture (%)": None if _summary["mean_top3_capture"] is None else round(100 * _summary["mean_top3_capture"], 1),
            "NDCG@3 (%)": None if _summary["mean_ndcg_at_3"] is None else round(100 * _summary["mean_ndcg_at_3"], 1),
        })
    mo.ui.table(pl.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What did the model learn?

    A positive weight favours higher values when the other feature stays the same. These weights apply after log-transforming earlier listens and scaling both features.
    """)
    return


@app.cell
def _(fold_results, mo, pl):
    _rows = []
    for _result in fold_results:
        _weights = _result.model.named_steps["classifier"].coef_[0]
        _positives = _result.fold.training.filter(pl.col("target_60d"))
        _rows.append({
            "Period": _result.fold.period,
            "Training rows": _result.fold.training_rows,
            "Breakouts represented in training": _positives.select("user_id", "artist_id").n_unique(),
            "Recent share weight": round(float(_weights[0]), 3),
            "Earlier listening weight": round(float(_weights[1]), 3),
        })
    mo.ui.table(pl.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(baseline_evaluation, mo, model_evaluation, paired_events):
    _baseline = baseline_evaluation.event_summary.row(0, named=True)
    _model = model_evaluation.event_summary.row(0, named=True)
    _counts = dict(paired_events.group_by("change").len().iter_rows())
    mo.md(f"""
    ## What do these results show?

    - Detected breakouts: **{_baseline['detected_events']} → {_model['detected_events']}**.
    - Anticipatory detections: **{_baseline['anticipatory_detections']} → {_model['anticipatory_detections']}**.
    - Among events both approaches detected: **{_counts.get('Earlier', 0)} earlier**, **{_counts.get('Later', 0)} later**, and **{_counts.get('Same day', 0)} unchanged**.
    - The model adds **{_counts.get('Newly detected', 0)}** detections and loses **{_counts.get('Lost detection', 0)}**.

    More anticipatory detections are useful only when considered alongside coverage losses and delays elsewhere. The period comparison shows whether the changes are spread through the history.
    """)
    return


@app.cell
def _(mo):
    record_button = mo.ui.run_button(label="Record comparison in MLflow")
    record_button
    return (record_button,)


@app.cell
def _(
    Path,
    baseline_evaluation,
    comparison_context,
    fold_results,
    mo,
    model_evaluation,
    record_button,
    record_comparison,
):
    mo.stop(not record_button.value)
    _run_id = record_comparison(
        baseline=baseline_evaluation,
        model=model_evaluation,
        fold_results=fold_results,
        context=comparison_context,
        project_root=Path("."),
    )
    mo.md(f"Recorded comparison: `{_run_id}`.")
    return


if __name__ == "__main__":
    app.run()
