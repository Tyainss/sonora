from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sonora.breakout import (
    aggregate_daily_listening,
    detect_first_breakouts,
    eligible_candidates_for_date,
    get_breakout_watchlist,
    make_daily_targets,
)
from sonora.breakout.watchlist import WATCHLIST_SCHEMA
from sonora.data.paths import DataPaths

BASE = date(2020, 1, 1)


def day(offset):
    return BASE + timedelta(days=offset)


def listen(offset, hour=12):
    return datetime.combine(day(offset), datetime.min.time(), tzinfo=UTC).replace(
        hour=hour
    )


def history_bounds(users=("u",), *, last_complete=80):
    return pl.DataFrame(
        [
            {
                "user_id": user,
                "first_date": day(0),
                "warmup_end": day(14),
                "score_start": day(35),
                "last_complete_date": day(last_complete),
            }
            for user in users
        ]
    )


def write_data(paths, rows):
    paths.curated_dir.mkdir(parents=True)
    events = pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "listened_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    events.write_parquet(paths.curated_listening_events)
    artists = sorted({row[1] for row in rows})
    pl.DataFrame(
        {"artist_id": artists, "canonical_name": [a.upper() for a in artists]}
    ).write_parquet(paths.curated_artists)


def simple_history():
    return [
        ("u", "first", listen(0)),
        *[("u", "a", listen(34, 20)) for _ in range(4)],
        *[("u", "b", listen(34, 19)) for _ in range(3)],
        *[("u", "c", listen(33, 18)) for _ in range(2)],
        ("u", "d", listen(34, 17)),
        *[("u", "a", listen(35, 20)) for _ in range(40)],
        ("u", "today_only", listen(35, 10)),
        ("v", "first", listen(0)),
        ("v", "a", listen(34, 22)),
        *[("v", "d", listen(34, 21)) for _ in range(3)],
    ]


def test_watchlist_top_three_and_point_in_time(tmp_path):
    paths = DataPaths(tmp_path)
    write_data(paths, simple_history())

    watchlist = get_breakout_watchlist(
        "u", day(35), user_bounds=history_bounds(("u", "v")), paths=paths
    )

    assert watchlist.schema == WATCHLIST_SCHEMA
    assert watchlist["artist_id"].to_list() == ["a", "b", "c"]
    assert watchlist["canonical_name"].to_list() == ["A", "B", "C"]
    assert watchlist["rank"].to_list() == [1, 2, 3]
    assert watchlist["listening_share_7d"].to_list() == pytest.approx(
        [0.4, 0.3, 0.2]
    )
    assert watchlist["artist_scrobbles_7d"].to_list() == [4, 3, 2]
    assert watchlist["user_scrobbles_7d"].to_list() == [10, 10, 10]
    assert watchlist["prior_scrobbles"].to_list() == [4, 3, 2]
    assert watchlist["latest_listened_at"].max() < listen(35, 0)

    other = get_breakout_watchlist(
        "v", day(35), user_bounds=history_bounds(("u", "v")), paths=paths
    )
    assert other["artist_id"].to_list()[:2] == ["d", "a"]
    assert other["listening_share_7d"].to_list()[:2] == pytest.approx([0.75, 0.25])
    assert set(other["user_id"]) == {"v"}


def test_different_future_events_do_not_change_historical_watchlist(tmp_path):
    paths = DataPaths(tmp_path)
    bounds = history_bounds(("u", "v"))
    write_data(paths, simple_history())
    original = get_breakout_watchlist("u", day(35), user_bounds=bounds, paths=paths)

    write_data(
        paths,
        [
            *simple_history(),
            *[("u", "b", listen(50)) for _ in range(200)],
        ],
    )
    assert_frame_equal(
        original, get_breakout_watchlist("u", day(35), user_bounds=bounds, paths=paths)
    )


def test_scoring_does_not_require_target_build_metadata(tmp_path):
    paths = DataPaths(tmp_path)
    bounds = history_bounds(("u", "v"))
    write_data(paths, simple_history())
    assert not paths.breakout_build_metadata.exists()

    watchlist = get_breakout_watchlist("u", day(35), user_bounds=bounds, paths=paths)
    assert watchlist["artist_id"].to_list() == ["a", "b", "c"]


def test_invalid_date_and_missing_history_coverage(tmp_path):
    paths = DataPaths(tmp_path)
    write_data(paths, simple_history())
    bounds = history_bounds(("u", "v"))
    with pytest.raises(ValueError, match="scoring start"):
        get_breakout_watchlist("u", day(34), user_bounds=bounds, paths=paths)
    with pytest.raises(ValueError, match="complete through D-1"):
        get_breakout_watchlist("u", day(82), user_bounds=bounds, paths=paths)
    with pytest.raises(TypeError, match="date"):
        get_breakout_watchlist("u", listen(35), user_bounds=bounds, paths=paths)
    with pytest.raises(ValueError, match="No history bounds"):
        get_breakout_watchlist("unknown", day(35), user_bounds=bounds, paths=paths)


def test_date_specific_eligibility_matches_historical_daily_targets():
    rows = [
        ("u", "return", listen(0)),
        ("u", "return", listen(40)),
        ("u", "return", listen(60)),
        ("u", "confirmed", listen(42)),
        ("u", "established", listen(44)),
        ("u", "same_day", listen(60)),
    ]
    listening = pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "listened_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    daily_user, daily_artist = aggregate_daily_listening(listening.lazy())
    bounds = history_bounds()
    events = pl.DataFrame(
        [("u", "confirmed", day(60))],
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "confirmation_date": pl.Date,
        },
        orient="row",
    )
    established = pl.DataFrame(
        [("u", "established")],
        schema={"user_id": pl.String, "artist_id": pl.String},
        orient="row",
    )
    candidates = eligible_candidates_for_date(
        daily_artist, events, established, bounds, day(60)
    )
    historical = (
        make_daily_targets(daily_artist, events, established, bounds)
        .collect()
        .filter(pl.col("scoring_date") == day(60))
        .select("user_id", "artist_id", "scoring_date")
        .sort("user_id", "artist_id")
    )
    assert_frame_equal(candidates, historical)
    assert candidates["artist_id"].to_list() == ["return"]
    assert daily_user.height > 0


def test_confirmed_breakout_drops_from_watchlist_on_confirmation_day(tmp_path):
    paths = DataPaths(tmp_path)
    rows = [
        ("u", f"background_{i}", listen(n))
        for i in range(10)
        for n in range(65)
    ]
    rows += [
        ("u", "target", listen(n, 16))
        for n in (45, 47, 52, 54)
        for _ in range(100)
    ]
    write_data(paths, rows)
    listening = pl.read_parquet(paths.curated_listening_events)
    daily_user, daily_artist = aggregate_daily_listening(
        listening.filter(pl.col("listened_at") < listen(55, 0)).lazy()
    )
    events, _ = detect_first_breakouts(
        daily_user,
        daily_artist,
        history_bounds(last_complete=54),
    )
    assert events.filter(pl.col("artist_id") == "target")[
        "confirmation_date"
    ].to_list() == [day(55)]

    before = get_breakout_watchlist(
        "u", day(54), user_bounds=history_bounds(), paths=paths
    )
    on_confirmation = get_breakout_watchlist(
        "u", day(55), user_bounds=history_bounds(), paths=paths
    )
    assert "target" in before["artist_id"].to_list()
    assert "target" not in on_confirmation["artist_id"].to_list()


def test_empty_watchlist_keeps_schema(tmp_path):
    paths = DataPaths(tmp_path)
    rows = [("u", "old", listen(0)), ("u", "old", listen(1))]
    write_data(paths, rows)
    watchlist = get_breakout_watchlist(
        "u", day(100), user_bounds=history_bounds(last_complete=120), paths=paths
    )
    assert watchlist.is_empty()
    assert watchlist.schema == WATCHLIST_SCHEMA


def test_recency_boundary_and_same_day_listens():
    daily_artist = pl.DataFrame(
        [
            ("u", "within_90", day(0), 1),
            ("u", "older", day(-1), 1),
            ("u", "today", day(90), 1),
        ],
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "date": pl.Date,
            "artist_scrobbles": pl.UInt32,
        },
        orient="row",
    )
    empty_events = pl.DataFrame(
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "confirmation_date": pl.Date,
        }
    )
    empty_established = pl.DataFrame(
        schema={"user_id": pl.String, "artist_id": pl.String}
    )
    bounds = history_bounds(last_complete=89).with_columns(
        pl.lit(day(-1)).cast(pl.Date).alias("first_date")
    )

    candidates = eligible_candidates_for_date(
        daily_artist, empty_events, empty_established, bounds, day(90)
    )

    assert candidates["artist_id"].to_list() == ["within_90"]
