import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sonora.breakout import build_modeling_data
from sonora.breakout.evaluation import EvaluationPeriod, build_walk_forward_folds

KEYS = ["user_id", "artist_id", "scoring_date"]
FEATURES = ["listening_share_7d", "previous_listening_count"]


def targets():
    return pl.DataFrame(
        {
            "user_id": ["u", "u", "v"],
            "artist_id": ["a", "a", "a"],
            "scoring_date": [
                dt.date(2022, 1, 10),
                dt.date(2022, 1, 2),
                dt.date(2022, 1, 10),
            ],
            "first_listen_date": [dt.date(2021, 12, 1)] * 3,
            "last_listen_date": [dt.date(2022, 1, 1)] * 3,
            "horizon_end_date": [
                dt.date(2022, 3, 11),
                dt.date(2022, 3, 3),
                dt.date(2022, 3, 11),
            ],
            "target_60d": [None, False, True],
            "has_full_horizon": [False, True, False],
        }
    )


def history():
    return pl.DataFrame(
        {
            "user_id": ["u", "u", "u", "v"],
            "artist_id": ["a", "a", "outside_candidates", "a"],
            "listened_at": [
                dt.datetime(2021, 12, 1, tzinfo=dt.UTC),
                dt.datetime(2022, 1, 1, tzinfo=dt.UTC),
                dt.datetime(2022, 1, 1, 12, tzinfo=dt.UTC),
                dt.datetime(2021, 12, 1, tzinfo=dt.UTC),
            ],
        }
    )


def test_modeling_data_preserves_targets_and_appends_raw_features_by_key():
    source = targets()
    result = build_modeling_data(history().lazy(), source.lazy())

    assert_frame_equal(result.select(source.columns), source)
    assert result.columns == [*source.columns, *FEATURES]
    assert result["listening_share_7d"].to_list() == [0.0, 0.5, 0.0]
    assert result["previous_listening_count"].to_list() == [2, 1, 1]
    assert result.schema["previous_listening_count"] == pl.UInt32


def test_features_do_not_depend_on_target_labels_or_future_observation_flags():
    source = targets()
    altered = source.with_columns(
        pl.lit(False).alias("target_60d"),
        pl.lit(True).alias("has_full_horizon"),
        pl.col("horizon_end_date") + pl.duration(days=10),
    )
    assert_frame_equal(
        build_modeling_data(history(), source).select(*KEYS, *FEATURES),
        build_modeling_data(history(), altered).select(*KEYS, *FEATURES),
    )


def test_walk_forward_folds_keep_features_and_enforce_existing_label_boundaries():
    dates = [
        dt.date(2021, 10, 1),
        dt.date(2021, 11, 2),
        dt.date(2021, 12, 15),
        dt.date(2022, 1, 1),
        dt.date(2022, 7, 1),
        dt.date(2022, 9, 1),
    ]
    source = pl.DataFrame(
        {
            "user_id": ["u"] * 6,
            "artist_id": ["old", "edge", "fast", "h1", "h2", "unknown"],
            "scoring_date": dates,
            "horizon_end_date": [day + dt.timedelta(days=60) for day in dates],
            "target_60d": [False, False, True, True, False, None],
            "has_full_horizon": [True] * 5 + [False],
        }
    )
    events = source.select("user_id", "artist_id").with_columns(
        pl.Series(
            "listened_at",
            [
                dt.datetime.combine(
                    day - dt.timedelta(days=1), dt.time(), tzinfo=dt.UTC
                )
                for day in dates
            ],
        )
    )
    periods = [
        EvaluationPeriod("2022 H1", dt.date(2022, 1, 1), dt.date(2022, 6, 30), False),
        EvaluationPeriod("2022 H2", dt.date(2022, 7, 1), dt.date(2022, 7, 1), True),
    ]
    data = build_modeling_data(events, source)
    first, second = build_walk_forward_folds(data, periods)
    assert first.training["artist_id"].to_list() == ["old"]
    assert second.training["artist_id"].to_list() == ["old", "edge", "fast", "h1"]
    assert data["target_60d"].null_count() == 1

    for expected, actual in zip(
        build_walk_forward_folds(source, periods), [first, second], strict=True
    ):
        assert_frame_equal(actual.training.select(source.columns), expected.training)
        assert_frame_equal(actual.scoring.select(source.columns), expected.scoring)
        for rows in [actual.training, actual.scoring]:
            assert rows["listening_share_7d"].to_list() == [1.0] * rows.height
            assert rows["previous_listening_count"].to_list() == [0] * rows.height


@pytest.mark.parametrize(
    "column", ["target_60d", "horizon_end_date", "has_full_horizon"]
)
def test_missing_label_metadata_is_rejected(column):
    with pytest.raises(ValueError, match="Missing target columns"):
        build_modeling_data(history(), targets().drop(column))


@pytest.mark.parametrize("column", FEATURES)
def test_existing_feature_columns_are_rejected(column):
    with pytest.raises(ValueError, match="already contain feature columns"):
        build_modeling_data(history(), targets().with_columns(pl.lit(0).alias(column)))


def test_duplicate_target_keys_are_rejected():
    source = targets()
    with pytest.raises(ValueError, match="duplicate rows"):
        build_modeling_data(history(), pl.concat([source, source.head(1)]))


def test_target_without_prior_listening_is_rejected():
    source = targets().with_columns(pl.lit("unheard").alias("artist_id"))
    with pytest.raises(ValueError, match="prior listening history"):
        build_modeling_data(history(), source)


def test_empty_targets_keep_their_schema_and_empty_feature_columns():
    source = targets().head(0)
    result = build_modeling_data(history().head(0), source)
    assert_frame_equal(result.select(source.columns), source)
    assert result.columns == [*source.columns, *FEATURES]
