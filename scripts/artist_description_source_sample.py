import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

_FAMILIAR_ARTISTS = (
    "IDLES",
    "Silvana Estrada",
    "Bruno Pernadas",
    "Linda Martini",
    "Niños del Cerro",
    "A Tribe Called Quest",
)
_GROUPS = ("1-2", "3-9", "10-29", "30-99", "100+")


def _listening_group(plays: int) -> str:
    if plays <= 2:
        return "1-2"
    if plays <= 9:
        return "3-9"
    if plays <= 29:
        return "10-29"
    if plays <= 99:
        return "30-99"
    return "100+"


def _choose_artists(rows: list[dict], per_group: int = 6) -> list[dict]:
    selected = []
    used_names = set()
    for group in _GROUPS:
        candidates = [
            row for row in rows if _listening_group(row["plays"]) == group
        ]
        candidates.sort(
            key=lambda row: (
                row["canonical_name"] not in _FAMILIAR_ARTISTS,
                hashlib.sha256(row["artist_id"].encode()).hexdigest(),
            )
        )
        group_selected = []
        for row in candidates:
            key = " ".join(row["canonical_name"].split()).casefold()
            if key in used_names:
                continue
            used_names.add(key)
            group_selected.append(
                {
                    "artist_id": row["artist_id"],
                    "artist_name": row["canonical_name"],
                    "plays": row["plays"],
                    "group": group,
                }
            )
            if len(group_selected) == per_group:
                break
        if len(group_selected) < per_group:
            raise ValueError(f"Not enough distinct artists in group {group}")
        selected.extend(group_selected)
    return selected


def _load_sample(artists_path: Path, events_path: Path) -> list[dict]:
    import polars as pl

    artists = pl.read_parquet(artists_path).select("artist_id", "canonical_name")
    plays = (
        pl.scan_parquet(events_path)
        .group_by("artist_id")
        .len(name="plays")
        .collect()
    )
    catalogue = artists.join(plays, on="artist_id", how="left").with_columns(
        pl.col("plays").fill_null(0)
    )
    return _choose_artists(list(catalogue.iter_rows(named=True)))


def _request_json(url: str, params: dict | None, user_agent: str) -> dict | None:
    endpoint = url
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected API response from {endpoint}")
    return payload


def _lastfm_description(name: str, api_key: str, user_agent: str) -> dict:
    response = _request_json(
        "https://ws.audioscrobbler.com/2.0/",
        {
            "method": "artist.getInfo",
            "artist": name,
            "api_key": api_key,
            "autocorrect": 0,
            "lang": "en",
            "format": "json",
        },
        user_agent,
    )
    if response is None or "error" in response:
        code = response.get("error") if response else None
        if code not in (6, 7, None):
            raise RuntimeError(f"Last.fm API error {code} for {name}")
        return {
            "status": "not_found" if code in (7, None) else "api_error",
            "response": response,
        }
    artist = response.get("artist")
    if not isinstance(artist, dict):
        raise ValueError(f"Unexpected Last.fm artist response for {name}")
    reported_name = artist.get("name")
    if not isinstance(reported_name, str):
        raise ValueError(f"Missing Last.fm artist name for {name}")
    bio = artist.get("bio") or {}
    if not isinstance(bio, dict):
        raise ValueError(f"Unexpected Last.fm biography for {name}")
    if " ".join(reported_name.split()).casefold() != " ".join(name.split()).casefold():
        status = "identity_mismatch"
    elif bio.get("summary") or bio.get("content"):
        status = "success"
    else:
        status = "empty"
    return {"status": status, "response": response}


def _wikipedia_candidates(name: str, user_agent: str) -> dict:
    search = _request_json(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        {"q": name, "limit": 5},
        user_agent,
    )
    if search is None or not isinstance(search.get("pages"), list):
        raise ValueError(f"Unexpected Wikipedia search response for {name}")
    candidates = []
    for page in search["pages"][:3]:
        title = page.get("title")
        if not isinstance(title, str):
            continue
        page_key = quote(title.replace(" ", "_"), safe="")
        summary = _request_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_key}",
            None,
            user_agent,
        )
        candidates.append({"search": page, "summary": summary})
    return {
        "status": "unverified" if candidates else "empty",
        "search_response": search,
        "candidates": candidates,
    }


def _read_ndjson(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def collect(
    *, out: Path, artists: Path, events: Path, user_agent: str, api_key: str
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sample_path = out / "sample.ndjson"
    result_path = out / "responses.ndjson"
    if sample_path.exists():
        sample = _read_ndjson(sample_path)
    else:
        if result_path.exists():
            raise ValueError("Results exist without a saved sample")
        sample = _load_sample(artists, events)
        with sample_path.open("x", encoding="utf-8") as file:
            for artist in sample:
                file.write(json.dumps(artist, ensure_ascii=False) + "\n")

    if len(sample) != 30 or len({row["artist_id"] for row in sample}) != 30:
        raise ValueError("Expected a saved sample of 30 distinct artists")
    existing = _read_ndjson(result_path)
    keys = [(record["artist_id"], record["source"]) for record in existing]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate results in the collection")
    known_ids = {row["artist_id"] for row in sample}
    if any(
        artist_id not in known_ids or source not in {"lastfm", "wikipedia"}
        for artist_id, source in keys
    ):
        raise ValueError("Results do not match the saved sample")
    completed = set(keys)
    with result_path.open("a", encoding="utf-8") as file:
        for index, artist in enumerate(sample, start=1):
            for source in ("lastfm", "wikipedia"):
                key = (artist["artist_id"], source)
                if key in completed:
                    continue
                result = (
                    _lastfm_description(artist["artist_name"], api_key, user_agent)
                    if source == "lastfm"
                    else _wikipedia_candidates(artist["artist_name"], user_agent)
                )
                record = {
                    "artist_id": artist["artist_id"],
                    "artist_name": artist["artist_name"],
                    "source": source,
                    "collected_at": datetime.now(UTC).isoformat(),
                    **result,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                completed.add(key)
            print(f"{index}/30: {artist['artist_name']}")
    print(f"Saved {len(completed)}/60 source responses in {result_path}")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare artist description sources on 30 artists"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/artist_description_source_sample")
    )
    parser.add_argument(
        "--artists", type=Path, default=Path("data/curated/artists.parquet")
    )
    parser.add_argument(
        "--events", type=Path, default=Path("data/curated/listening_events.parquet")
    )
    args = parser.parse_args()
    api_key = os.environ.get("LASTFM_API_KEY")
    user_agent = os.environ.get("WIKIMEDIA_USER_AGENT")
    if not api_key or not user_agent:
        parser.error(
            "Set LASTFM_API_KEY and WIKIMEDIA_USER_AGENT "
            "in your environment or .env"
        )
    collect(
        out=args.out,
        artists=args.artists,
        events=args.events,
        user_agent=user_agent,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
