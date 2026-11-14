import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "artist_description_source_sample.py"
)
_spec = importlib.util.spec_from_file_location("artist_description_source_sample", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_choose_artists = _module._choose_artists
_lastfm_description = _module._lastfm_description
_wikipedia_candidates = _module._wikipedia_candidates
collect = _module.collect


def test_sample_is_deterministic_and_includes_familiar_artists():
    rows = [
        {
            "artist_id": f"{band}-{index}",
            "canonical_name": f"Artist {band}-{index}",
            "plays": plays,
        }
        for band, plays in enumerate((1, 5, 15, 50, 120))
        for index in range(10)
    ]
    rows[0]["canonical_name"] = "IDLES"
    first = _choose_artists(rows)
    assert first == _choose_artists(list(reversed(rows)))
    assert len(first) == 30
    assert len({row["artist_name"] for row in first}) == 30
    assert "IDLES" in [row["artist_name"] for row in first]
    assert {
        group: sum(row["group"] == group for row in first)
        for group in ("1-2", "3-9", "10-29", "30-99", "100+")
    } == {
        group: 6 for group in ("1-2", "3-9", "10-29", "30-99", "100+")
    }


def test_lastfm_bio_and_identity_are_checked(monkeypatch):
    monkeypatch.setattr(
        _module, "_request_json",
        lambda *args: {
            "artist": {"name": "IDLES", "bio": {"summary": "Post-punk band"}}
        },
    )
    assert _lastfm_description("IDLES", "key", "agent")["status"] == "success"
    assert (
        _lastfm_description("Other artist", "key", "agent")["status"]
        == "identity_mismatch"
    )


def test_wikipedia_preserves_unverified_candidates(monkeypatch):
    def fake_request(url, params, user_agent):
        if "search/page" in url:
            return {"pages": [{"title": "IDLES", "description": "English band"}]}
        return {"title": "IDLES", "extract": "English punk rock band"}

    monkeypatch.setattr(
        _module, "_request_json", fake_request
    )
    result = _wikipedia_candidates("IDLES", "agent")
    assert result["status"] == "unverified"
    assert result["search_response"]["pages"][0]["title"] == "IDLES"
    assert (
        result["candidates"][0]["summary"]["extract"]
        == "English punk rock band"
    )


def test_resume_skips_completed_sources(tmp_path, monkeypatch):
    sample = [
        {
            "artist_id": f"artist-{i}",
            "artist_name": f"Artist {i}",
            "plays": 10,
            "group": "10-29",
        }
        for i in range(30)
    ]
    monkeypatch.setattr(
        _module, "_load_sample", lambda *args: sample
    )
    calls = []
    monkeypatch.setattr(
        _module, "_lastfm_description",
        lambda name, *args: (
            calls.append(("lastfm", name)) or {"status": "empty", "response": {}}
        ),
    )
    monkeypatch.setattr(
        _module, "_wikipedia_candidates",
        lambda name, *args: (
            calls.append(("wikipedia", name))
            or {"status": "empty", "search_response": {}, "candidates": []}
        ),
    )
    for _ in range(2):
        collect(
            out=tmp_path,
            artists=tmp_path / "artists",
            events=tmp_path / "events",
            user_agent="test",
            api_key="key",
        )
    assert len(calls) == 60


def test_resume_after_interruption_preserves_completed_source(tmp_path, monkeypatch):
    sample = [
        {"artist_id": str(i), "artist_name": f"Artist {i}", "plays": 1, "group": "1-2"}
        for i in range(30)
    ]
    monkeypatch.setattr(
        _module, "_load_sample", lambda *args: sample
    )
    calls = {"lastfm": 0, "wikipedia": 0}

    def lastfm(*args):
        calls["lastfm"] += 1
        return {"status": "success", "response": {}}

    def wikipedia(*args):
        calls["wikipedia"] += 1
        if calls["wikipedia"] == 1:
            raise RuntimeError("Temporary API failure")
        return {"status": "empty", "candidates": []}

    monkeypatch.setattr(
        _module, "_lastfm_description", lastfm
    )
    monkeypatch.setattr(
        _module, "_wikipedia_candidates", wikipedia
    )
    kwargs = {
        "out": tmp_path,
        "artists": tmp_path / "artists",
        "events": tmp_path / "events",
        "user_agent": "test",
        "api_key": "key",
    }
    with pytest.raises(RuntimeError, match="Temporary"):
        collect(**kwargs)
    collect(**kwargs)
    assert calls == {"lastfm": 30, "wikipedia": 31}
