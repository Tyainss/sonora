import datetime as dt
import json

import pytest

from sonora.breakout.evaluation.holdout import load_or_create_holdout


def test_holdout_stays_fixed_when_more_history_arrives(tmp_path):
    path = tmp_path / "first_ml.json"
    first = load_or_create_holdout(
        path,
        user_id="u",
        generation="v1",
        observation_coverage_end=dt.date(2026, 8, 10),
    )
    original = path.read_bytes()
    refreshed = load_or_create_holdout(
        path, user_id="u", generation="v1", observation_coverage_end=dt.date(2027, 2, 1)
    )
    assert refreshed == first
    assert path.read_bytes() == original
    assert first.event_start == dt.date(2025, 12, 12)
    assert first.event_end == dt.date(2026, 6, 11)
    assert first.development_end == dt.date(2025, 10, 12)


@pytest.mark.parametrize("user,generation", [("other", "v1"), ("u", "v2")])
def test_other_users_or_cycles_cannot_reuse_or_replace_the_manifest(
    tmp_path, user, generation
):
    path = tmp_path / "holdout.json"
    coverage = dt.date(2026, 8, 10)
    load_or_create_holdout(
        path, user_id="u", generation="v1", observation_coverage_end=coverage
    )
    original = path.read_bytes()
    with pytest.raises(ValueError, match="different user or generation"):
        load_or_create_holdout(
            path, user_id=user, generation=generation, observation_coverage_end=coverage
        )
    assert path.read_bytes() == original


def test_incomplete_coverage_or_inconsistent_boundaries_are_rejected(tmp_path):
    path = tmp_path / "holdout.json"
    load_or_create_holdout(
        path,
        user_id="u",
        generation="v1",
        observation_coverage_end=dt.date(2026, 8, 10),
    )
    with pytest.raises(ValueError, match="does not cover"):
        load_or_create_holdout(
            path,
            user_id="u",
            generation="v1",
            observation_coverage_end=dt.date(2026, 5, 1),
        )
    saved = json.loads(path.read_text())
    saved["boundaries"]["development_end"] = "2026-01-01"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="inconsistent"):
        load_or_create_holdout(
            path,
            user_id="u",
            generation="v1",
            observation_coverage_end=dt.date(2026, 8, 10),
        )


@pytest.mark.parametrize(
    "latest_end",
    [
        dt.date(2024, 2, 28),
        dt.date(2024, 2, 29),
        dt.date(2024, 8, 30),
        dt.date(2024, 8, 31),
    ],
)
def test_calendar_edges_keep_the_entire_holdout_observable(tmp_path, latest_end):
    holdout = load_or_create_holdout(
        tmp_path / "holdout.json",
        user_id="u",
        generation="v1",
        observation_coverage_end=latest_end + dt.timedelta(days=60),
    )
    assert holdout.event_end <= latest_end
    assert (latest_end - holdout.event_end).days <= 3
    assert (holdout.lead_in_end - holdout.lead_in_start).days == 59
    assert holdout.development_end < holdout.lead_in_start < holdout.event_start
