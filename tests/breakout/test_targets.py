import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sonora.breakout import detect_first_breakouts, make_daily_targets

BASE = dt.date(2020, 1, 1)


def day(n):
    return BASE + dt.timedelta(days=n)


def bounds(users=("a", "b"), end=260):
    return pl.DataFrame(
        [
            {
                "user_id": u,
                "first_date": day(0),
                "warmup_end": day(14),
                "score_start": day(35),
                "last_complete_date": day(end),
            }
            for u in users
        ]
    )


def artist_days(rows):
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "date": pl.Date,
            "artist_scrobbles": pl.UInt32,
        },
        orient="row",
    )


def event_rows(rows=()):
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "confirmation_date": pl.Date,
        },
        orient="row",
    )


def established_rows(rows=()):
    return pl.DataFrame(
        rows, schema={"user_id": pl.String, "artist_id": pl.String}, orient="row"
    )


def test_daily_rows_match_independent_replay():
    listens = [("a", "return", day(i), 1) for i in [0, 40, 180, 250]]
    listens += [
        ("b", "return", day(60), 1),
        ("a", "confirmed", day(40), 1),
        ("a", "known_recent", day(220), 1),
        ("a", "future", day(220), 1),
        ("a", "established", day(40), 1),
    ]
    events = event_rows(
        [
            ("a", "confirmed", day(101)),
            ("a", "known_recent", day(260)),
            ("a", "future", day(280)),
        ]
    )
    actual = make_daily_targets(
        artist_days(listens), events, established_rows([("a", "established")]), bounds()
    ).collect()
    expected = []
    confirmations = {
        (r["user_id"], r["artist_id"]): r["confirmation_date"]
        for r in events.to_dicts()
    }
    for user in ["a", "b"]:
        for n in range(35, 262):
            date = day(n)
            for artist in {r[1] for r in listens if r[0] == user}:
                earlier = sorted(
                    r[2]
                    for r in listens
                    if r[0] == user and r[1] == artist and r[2] < date
                )
                confirmation = confirmations.get((user, artist))
                if confirmation and confirmation > day(261):
                    confirmation = None
                if (
                    (user, artist) == ("a", "established")
                    or not earlier
                    or (date - earlier[-1]).days > 90
                    or (confirmation and confirmation <= date)
                ):
                    continue
                full = n + 60 <= 261
                target = (
                    True
                    if confirmation and date < confirmation <= day(n + 60)
                    else (False if full else None)
                )
                expected.append(
                    (
                        user,
                        artist,
                        date,
                        earlier[0],
                        earlier[-1],
                        day(n + 60),
                        target,
                        full,
                    )
                )
    expected = pl.DataFrame(expected, schema=actual.schema, orient="row").sort(
        "user_id", "scoring_date", "artist_id"
    )
    assert_frame_equal(actual, expected)
    returning = actual.filter(
        (pl.col("user_id") == "a") & (pl.col("artist_id") == "return")
    )
    assert day(130) in returning["scoring_date"]
    assert day(131) not in returning["scoring_date"]
    assert day(180) not in returning["scoring_date"]
    assert returning.filter(pl.col("scoring_date") == day(181))["first_listen_date"][
        0
    ] == day(0)
    confirmed = actual.filter(pl.col("artist_id") == "confirmed")
    assert confirmed.filter(pl.col("scoring_date") == day(41))["target_60d"][0] is True
    assert day(101) not in confirmed["scoring_date"]


def test_more_history_keeps_full_labels_and_resolves_unknowns():
    listens = artist_days([("a", "x", day(40), 1), ("a", "x", day(170), 1)])
    events = event_rows([("a", "x", day(220))])
    before = make_daily_targets(
        listens, events, established_rows(), bounds(("a",), 200)
    ).collect()
    after = make_daily_targets(
        listens, events, established_rows(), bounds(("a",), 260)
    ).collect()
    complete = before.filter(pl.col("has_full_horizon"))
    assert_frame_equal(
        complete,
        after.join(
            complete.select("user_id", "artist_id", "scoring_date"),
            on=["user_id", "artist_id", "scoring_date"],
            how="semi",
        ),
    )
    assert before.filter(pl.col("scoring_date") == day(180))["target_60d"][0] is None
    assert after.filter(pl.col("scoring_date") == day(180))["target_60d"][0] is True


def synthetic_history():
    rows = [
        (u, str(a), day(n), 1)
        for u in ["a", "b"]
        for a in range(10)
        for n in range(101)
    ]
    rows += [("a", "target", day(n), 100) for n in [45, 47, 52, 54, 70, 72, 77, 79]]
    rows += [("b", "target", day(n), 100) for n in [45, 52]]
    artists = artist_days(rows)
    users = artists.group_by("user_id", "date").agg(
        pl.col("artist_scrobbles").sum().alias("user_scrobbles")
    )
    return users, artists


def history_with_target_days(target_days):
    rows = [("a", str(a), day(n), 1) for a in range(10) for n in range(101)]
    rows += [("a", "target", day(n), 100) for n in target_days]
    artists = artist_days(rows)
    users = artists.group_by("user_id", "date").agg(
        pl.col("artist_scrobbles").sum().alias("user_scrobbles")
    )
    return users, artists


def test_completed_second_period_confirms_immediately_and_only_once():
    users, artists = synthetic_history()
    events, _ = detect_first_breakouts(users, artists, bounds(end=54))
    target = events.filter(pl.col("artist_id") == "target")
    assert target["user_id"].to_list() == ["a"]
    assert target["confirmation_date"].to_list() == [day(55)]
    assert target["breakout_period_start"].to_list() == [day(41)]
    before, _ = detect_first_breakouts(users, artists, bounds(end=53))
    assert before.filter(pl.col("artist_id") == "target").is_empty()
    later, _ = detect_first_breakouts(users, artists, bounds(end=100))
    assert_frame_equal(target, later.filter(pl.col("artist_id") == "target"))
    permissive, _ = detect_first_breakouts(
        users, artists, bounds(end=54), min_active_days=1
    )
    assert set(permissive.filter(pl.col("artist_id") == "target")["user_id"]) == {
        "a",
        "b",
    }


def test_warmup_established_artist_never_gets_later_first_breakout():
    users, artists = history_with_target_days([1, 3, 8, 10, 45, 47, 52, 54])

    events, established = detect_first_breakouts(users, artists, bounds(("a",), 100))

    assert established.filter(pl.col("artist_id") == "target").height == 1
    assert events.filter(pl.col("artist_id") == "target").is_empty()


def test_repeated_periods_require_non_overlap_and_allow_up_to_14_day_gap():
    overlapping_users, overlapping_artists = history_with_target_days([45, 47, 48])
    edge_users, edge_artists = history_with_target_days([45, 47, 59, 61])
    late_users, late_artists = history_with_target_days([45, 47, 64, 66])

    overlapping, _ = detect_first_breakouts(
        overlapping_users, overlapping_artists, bounds(("a",), 80)
    )
    edge, _ = detect_first_breakouts(edge_users, edge_artists, bounds(("a",), 80))
    late, _ = detect_first_breakouts(late_users, late_artists, bounds(("a",), 80))

    assert overlapping.filter(pl.col("artist_id") == "target").is_empty()
    assert edge.filter(pl.col("artist_id") == "target").height == 1
    assert late.filter(pl.col("artist_id") == "target").is_empty()


def test_empty_and_short_histories_keep_output_types():
    users, artists = synthetic_history()
    events, established = detect_first_breakouts(
        users.head(0), artists.head(0), bounds(end=10)
    )
    result = make_daily_targets(artists, events, established, bounds(end=10)).collect()
    assert result.is_empty()
    assert result.schema["target_60d"] == pl.Boolean
    assert result.schema["scoring_date"] == pl.Date


def test_invalid_settings_and_duplicate_bounds_fail():
    users, artists = synthetic_history()
    with pytest.raises(ValueError):
        detect_first_breakouts(users, artists, pl.concat([bounds(), bounds()]))
    with pytest.raises(ValueError):
        make_daily_targets(
            artists, event_rows(), established_rows(), bounds(), horizon_days=30
        )
    with pytest.raises(ValueError):
        make_daily_targets(
            artists, event_rows(), established_rows(), bounds(), recency_days=0
        )


def test_horizon_edges_and_confirmation_day_removal():
    listens = artist_days(
        [("a", a, day(39), 1) for a in ["tomorrow", "edge", "outside", "today"]]
    )
    events = event_rows(
        [
            ("a", "tomorrow", day(41)),
            ("a", "edge", day(100)),
            ("a", "outside", day(101)),
            ("a", "today", day(40)),
        ]
    )
    rows = (
        make_daily_targets(listens, events, established_rows(), bounds(("a",)))
        .collect()
        .filter(pl.col("scoring_date") == day(40))
    )
    assert dict(rows.select("artist_id", "target_60d").iter_rows()) == {
        "tomorrow": True,
        "edge": True,
        "outside": False,
    }


def test_larger_recency_keeps_every_existing_day():
    listens = artist_days([("a", "x", day(40), 1), ("a", "x", day(180), 1)])
    previous = set()
    for recency in [30, 60, 90]:
        rows = make_daily_targets(
            listens,
            event_rows(),
            established_rows(),
            bounds(("a",)),
            recency_days=recency,
        ).collect()
        current = set(rows["scoring_date"])
        assert previous <= current
        previous = current
