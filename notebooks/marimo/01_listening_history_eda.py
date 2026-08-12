import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Listening history EDA

    A first look at the Last.fm listening history, focused on data quality, identity issues, and patterns that may matter for breakout later.

    The data comes from one user, but Sonora should eventually work for many users, so the checks focus on problems we can expect to see again.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import polars as pl

    DATA_PATH = Path("data/raw/lastfm/scrobbles.ndjson")
    return DATA_PATH, alt, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Raw history quality
    """)
    return


@app.cell
def _(DATA_PATH, pl):
    scrobbles = pl.scan_ndjson(DATA_PATH)

    scrobbles.collect_schema()
    return (scrobbles,)


@app.cell
def _(pl, scrobbles):
    summary = (
        scrobbles
        .with_columns(
            pl.from_epoch("timestamp_unix", time_unit="s")
            .dt.replace_time_zone("UTC")
            .alias("timestamp")
        )
        .select(
            pl.len().alias("scrobbles"),
            pl.col("timestamp").min().alias("first_scrobble"),
            pl.col("timestamp").max().alias("last_scrobble"),
        )
        .collect()
    )

    summary
    return


@app.cell
def _(pl, scrobbles):
    counts = (
        scrobbles
        .select(
            pl.len().alias("scrobbles"),
            pl.col("artist_name").n_unique().alias("artists"),
            pl.struct("artist_name", "track_name").n_unique().alias("tracks"),
            pl.struct("artist_name", "album_name").n_unique().alias("albums"),
        )
        .collect()
    )

    counts
    return (counts,)


@app.cell
def _(pl, scrobbles):
    missing = (
        scrobbles
        .select(
            pl.col("artist_name").is_null().sum().alias("artist_nulls"),
            pl.col("track_name").is_null().sum().alias("track_nulls"),
            pl.col("album_name").is_null().sum().alias("album_nulls"),
            (pl.col("artist_name") == "").sum().alias("artist_empty"),
            (pl.col("track_name") == "").sum().alias("track_empty"),
            (pl.col("album_name") == "").sum().alias("album_empty"),
        )
        .collect()
    )

    missing
    return


@app.cell
def _(pl, scrobbles):
    listening = scrobbles.with_columns(
        pl.from_epoch("timestamp_unix", time_unit="s")
        .dt.replace_time_zone("UTC")
        .alias("timestamp")
    )
    return (listening,)


@app.cell
def _(listening, pl):
    monthly = (
        listening
        .with_columns(
            pl.col("timestamp").dt.truncate("1mo").alias("month")
        )
        .group_by("month")
        .agg(
            pl.len().alias("scrobbles")
        )
        .sort("month")
        .collect()
    )
    return (monthly,)


@app.cell
def _(alt, monthly):
    alt.Chart(monthly).mark_line().encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("scrobbles:Q", title="Scrobbles"),
    )
    return


@app.cell
def _(pl, scrobbles):
    identifier_coverage = (
        scrobbles
        .select(
            pl.len().alias("scrobbles"),
            pl.col("mbid").is_not_null().sum().alias("track_mbid"),
            (
                pl.col("raw")
                .struct.field("artist")
                .struct.field("mbid")
                .ne("")
                .sum()
            ).alias("artist_mbid"),
            (
                pl.col("raw")
                .struct.field("album")
                .struct.field("mbid")
                .ne("")
                .sum()
            ).alias("album_mbid"),
        )
        .with_columns(
            (pl.col("track_mbid") / pl.col("scrobbles") * 100)
            .round(1)
            .alias("track_mbid_pct"),
            (pl.col("artist_mbid") / pl.col("scrobbles") * 100)
            .round(1)
            .alias("artist_mbid_pct"),
            (pl.col("album_mbid") / pl.col("scrobbles") * 100)
            .round(1)
            .alias("album_mbid_pct"),
        )
        .collect()
    )

    identifier_coverage
    return


@app.cell
def _(listening, pl):
    track_mbid_by_year = (
        listening
        .with_columns(
            pl.col("timestamp").dt.year().alias("year")
        )
        .group_by("year")
        .agg(
            pl.len().alias("scrobbles"),
            pl.col("mbid").is_not_null().sum().alias("with_mbid"),
        )
        .with_columns(
            (
                pl.col("with_mbid")
                / pl.col("scrobbles")
                * 100
            )
            .round(1)
            .alias("mbid_pct")
        )
        .sort("year")
        .collect()
    )

    track_mbid_by_year
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The history contains **177,271 scrobbles** from March 2020 to August 2026, with no missing artist, track, or album names. Every month is represented.

    Listening was highest in 2020-2021, dropped in 2022, and has stayed lower since. August 2026 is still a partial month.

    MBIDs are available for **74.3% of tracks, 80.9% of artists, and 76.1% of albums**. Track coverage stays fairly similar across years, so the missing IDs are spread throughout the history.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Artist and track identity
    """)
    return


@app.cell
def _(pl, scrobbles):
    artist_name_variants = (
        scrobbles
        .group_by("artist_name")
        .agg(
            pl.len().alias("scrobbles")
        )
        .with_columns(
            (
                pl.col("artist_name")
                .str.normalize("NFKD")
                .str.to_lowercase()
                .str.replace_all(r"\p{M}", "")
                .str.replace_all(r"[^\p{L}\p{N}]+", "")
            ).alias("comparison_key")
        )
        .group_by("comparison_key")
        .agg(
            pl.len().alias("name_count"),
            pl.col("artist_name").alias("artist_names"),
            pl.col("scrobbles").alias("scrobble_counts"),
            pl.col("scrobbles").sum().alias("total_scrobbles"),
        )
        .filter(pl.col("name_count") > 1)
        .sort("total_scrobbles", descending=True)
        .collect()
    )

    artist_name_variants
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simple formatting differences already split some artists into separate names. The check found **10 groups**, including `Samuel Uria` / `Samuel Úria`, several `JAY-Z` spellings, and `Luisa Sobral` / `Luísa Sobral`.
    """)
    return


@app.cell
def _(pl, scrobbles):
    artist_tracks = (
        scrobbles
        .select("artist_name", "track_name")
        .unique()
    )

    shared_track_artist_pairs = (
        artist_tracks
        .join(
            artist_tracks,
            on="track_name",
            how="inner",
            suffix="_other",
        )
        .filter(
            pl.col("artist_name") < pl.col("artist_name_other")
        )
        .group_by("artist_name", "artist_name_other")
        .agg(
            pl.len().alias("shared_tracks"),
            pl.col("track_name").head(5).alias("examples"),
        )
        .filter(pl.col("shared_tracks") >= 2)
        .sort("shared_tracks", descending=True)
        .collect()
    )

    shared_track_artist_pairs.head(30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looking at shared tracks finds a few more cases that formatting alone misses, such as `Conjunto Corona` / `Corona`, `Miguel Luz` / `Mike Lyte`, and `Aoba Ichiko` / `Ichiko Aoba`. Many other pairs are simply different artists with tracks that happen to share a title.
    """)
    return


@app.cell
def _(pl, scrobbles):
    track_name_variants = (
        scrobbles
        .group_by("artist_name", "track_name")
        .agg(
            pl.len().alias("scrobbles")
        )
        .with_columns(
            (
                pl.col("track_name")
                .str.normalize("NFKD")
                .str.to_lowercase()
                .str.replace_all(r"\p{M}", "")
                .str.replace_all(r"[^\p{L}\p{N}]+", "")
            ).alias("track_comparison_key")
        )
        .group_by("artist_name", "track_comparison_key")
        .agg(
            pl.len().alias("name_count"),
            pl.col("track_name").alias("track_names"),
            pl.col("scrobbles").alias("scrobble_counts"),
            pl.col("scrobbles").sum().alias("total_scrobbles"),
        )
        .filter(pl.col("name_count") > 1)
        .sort("total_scrobbles", descending=True)
        .collect()
    )

    track_name_variants.head(30)
    return (track_name_variants,)


@app.cell
def _(counts, pl, track_name_variants):
    track_variant_summary = pl.DataFrame(
        {
            "metric": [
                "artist-track pairs",
                "possible variant groups",
                "track names in those groups",
                "scrobbles in those groups",
            ],
            "value": [
                counts["tracks"][0],
                track_name_variants.height,
                track_name_variants["name_count"].sum(),
                track_name_variants["total_scrobbles"].sum(),
            ],
        }
    )

    track_variant_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Track names have similar inconsistencies. The check found **74 possible variant groups**, covering **1,321 scrobbles** (about 0.7% of the history). Most are small punctuation, spacing, or accent differences.

    Normalising names too aggressively can also merge different tracks, such as `Change` and `Chang'e`. `artist_name + track_name` is still a useful starting point, but the shared data layer will need to handle these cases consistently across users. MBIDs can help where available.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Listening behaviour over time
    """)
    return


@app.cell
def _(listening, pl):
    monthly_artist_activity = (
        listening
        .with_columns(
            pl.col("timestamp").dt.truncate("1mo").alias("month")
        )
        .group_by("month", "artist_name")
        .agg(
            pl.len().alias("artist_scrobbles")
        )
        .with_columns(
            pl.col("artist_scrobbles")
            .rank("ordinal", descending=True)
            .over("month")
            .alias("artist_rank")
        )
    )

    monthly_active_days = (
        listening
        .with_columns(
            pl.col("timestamp").dt.truncate("1mo").alias("month"),
            pl.col("timestamp").dt.date().alias("date"),
        )
        .group_by("month")
        .agg(
            pl.col("date").n_unique().alias("active_days")
        )
    )

    monthly_behavior = (
        monthly_artist_activity
        .group_by("month")
        .agg(
            pl.col("artist_scrobbles").sum().alias("scrobbles"),
            pl.len().alias("active_artists"),
            (
                pl.when(pl.col("artist_rank") <= 10)
                .then(pl.col("artist_scrobbles"))
                .otherwise(0)
                .sum()
            ).alias("top_10_scrobbles"),
        )
        .join(monthly_active_days, on="month", how="left")
        .with_columns(
            (
                pl.col("scrobbles") / pl.col("active_days")
            )
            .round(1)
            .alias("scrobbles_per_active_day"),
            (
                pl.col("top_10_scrobbles") / pl.col("scrobbles") * 100
            )
            .round(1)
            .alias("top_10_artist_share_pct"),
        )
        .sort("month")
        .collect()
    )
    return (monthly_behavior,)


@app.cell
def _(alt, monthly_behavior):
    intensity_chart = (
        alt.Chart(monthly_behavior)
        .mark_line()
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y(
                "scrobbles_per_active_day:Q",
                title="Scrobbles / active day",
                scale=alt.Scale(zero=False),
            ),
        )
        .properties(height=130)
    )

    breadth_chart = (
        alt.Chart(monthly_behavior)
        .mark_line()
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y(
                "active_artists:Q",
                title="Active artists",
                scale=alt.Scale(zero=False),
            ),
        )
        .properties(height=130)
    )

    concentration_chart = (
        alt.Chart(monthly_behavior)
        .mark_line()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y(
                "top_10_artist_share_pct:Q",
                title="Top 10 share (%)",
                scale=alt.Scale(zero=False),
            ),
        )
        .properties(height=130)
    )

    alt.vconcat(
        intensity_chart,
        breadth_chart,
        concentration_chart,
        spacing=10,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Listening behaviour changes noticeably over time. January 2021 stands out with **162 scrobbles per active day** and **563 active artists**, while quieter months can fall to around 45-60 scrobbles per active day and fewer than 200 active artists.

    Concentration changes too: the top 10 artists account for roughly **21% to 66%** of monthly listening. Later breakout logic will therefore need to adapt to each user's listening level rather than rely mainly on fixed play counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Artist adoption patterns
    """)
    return


@app.cell
def _(listening, pl):
    artist_active_days = (
        listening
        .select(
            "artist_name",
            pl.col("timestamp").dt.date().alias("date"),
        )
        .unique()
        .with_columns(
            pl.col("date")
            .shift(1)
            .over("artist_name", order_by="date")
            .alias("previous_active_date")
        )
        .with_columns(
            (
                pl.col("date") - pl.col("previous_active_date")
            )
            .dt.total_days()
            .alias("gap_days")
        )
    )

    artist_history = (
        listening
        .with_columns(
            pl.col("timestamp").dt.date().alias("date")
        )
        .group_by("artist_name")
        .agg(
            pl.len().alias("scrobbles"),
            pl.col("date").n_unique().alias("active_days"),
            pl.col("timestamp").min().alias("first_seen"),
            pl.col("timestamp").max().alias("last_seen"),
        )
        .join(
            artist_active_days
            .group_by("artist_name")
            .agg(
                pl.col("gap_days").max().alias("max_gap_days")
            ),
            on="artist_name",
            how="left",
        )
    )
    return artist_active_days, artist_history


@app.cell
def _(artist_history, pl):
    first_observed_by_month = (
        artist_history
        .with_columns(
            pl.col("first_seen").dt.truncate("1mo").alias("month")
        )
        .group_by("month")
        .agg(
            pl.len().alias("first_observed_artists")
        )
        .sort("month")
        .collect()
    )
    return (first_observed_by_month,)


@app.cell
def _(alt, first_observed_by_month):
    alt.Chart(first_observed_by_month).mark_bar().encode(
        x=alt.X("month:T", title="First observed month"),
        y=alt.Y("first_observed_artists:Q", title="Artists"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Many artist names first appear near the beginning of the history, when existing listening relationships are being seen for the first time. After that, the number of first-seen artists is usually much lower, with occasional spikes.
    """)
    return


@app.cell
def _(artist_history, listening, pl):
    first_seen_dates = artist_history.select(
        "artist_name",
        pl.col("first_seen").dt.date().alias("first_seen_date"),
    )

    first_day_engagement = (
        listening
        .with_columns(
            pl.col("timestamp").dt.date().alias("date")
        )
        .join(first_seen_dates, on="artist_name", how="inner")
        .filter(pl.col("date") == pl.col("first_seen_date"))
        .group_by("artist_name")
        .agg(
            pl.len().alias("first_day_scrobbles"),
            pl.col("track_name").n_unique().alias("first_day_tracks"),
        )
    )

    first_day_summary = (
        first_day_engagement
        .select(
            pl.len().alias("artists"),
            pl.col("first_day_scrobbles").median().alias("median_first_day_scrobbles"),
            pl.col("first_day_tracks").median().alias("median_first_day_tracks"),
            (pl.col("first_day_scrobbles") == 1).sum().alias("one_scrobble_artists"),
            (pl.col("first_day_tracks") == 1).sum().alias("one_track_artists"),
        )
        .with_columns(
            (
                pl.col("one_scrobble_artists") / pl.col("artists") * 100
            )
            .round(1)
            .alias("one_scrobble_pct"),
            (
                pl.col("one_track_artists") / pl.col("artists") * 100
            )
            .round(1)
            .alias("one_track_pct"),
        )
        .collect()
    )

    first_day_summary
    return


@app.cell
def _(artist_history, pl):
    artist_engagement_bands = (
        artist_history
        .with_columns(
            pl.when(pl.col("active_days") == 1)
            .then(pl.lit("1 day"))
            .when(pl.col("active_days") <= 3)
            .then(pl.lit("2-3 days"))
            .when(pl.col("active_days") <= 10)
            .then(pl.lit("4-10 days"))
            .otherwise(pl.lit("11+ days"))
            .alias("active_day_band"),
            pl.when(pl.col("active_days") == 1)
            .then(1)
            .when(pl.col("active_days") <= 3)
            .then(2)
            .when(pl.col("active_days") <= 10)
            .then(3)
            .otherwise(4)
            .alias("band_order"),
        )
        .group_by("active_day_band", "band_order")
        .agg(
            pl.len().alias("artists")
        )
        .with_columns(
            (
                pl.col("artists") / pl.col("artists").sum() * 100
            )
            .round(1)
            .alias("artist_pct")
        )
        .sort("band_order")
        .drop("band_order")
        .collect()
    )

    artist_engagement_bands
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most artists start with very light listening. The median first day contains **1 scrobble from 1 track**; **63.5%** of artists have only one scrobble that day, and **71.4%** only one track.

    The pattern continues over the full history: **38.9%** of artists appear on only one day, and **59.2%** appear on at most three days. Brief sampling is very common.
    """)
    return


@app.cell
def _(artist_history, pl):
    return_gap_bands = (
        artist_history
        .filter(pl.col("active_days") > 1)
        .with_columns(
            pl.when(pl.col("max_gap_days") < 30)
            .then(pl.lit("1-29 days"))
            .when(pl.col("max_gap_days") < 90)
            .then(pl.lit("30-89 days"))
            .when(pl.col("max_gap_days") < 180)
            .then(pl.lit("90-179 days"))
            .when(pl.col("max_gap_days") < 365)
            .then(pl.lit("180-364 days"))
            .otherwise(pl.lit("365+ days"))
            .alias("maximum_gap"),
            pl.when(pl.col("max_gap_days") < 30)
            .then(1)
            .when(pl.col("max_gap_days") < 90)
            .then(2)
            .when(pl.col("max_gap_days") < 180)
            .then(3)
            .when(pl.col("max_gap_days") < 365)
            .then(4)
            .otherwise(5)
            .alias("band_order"),
        )
        .group_by("maximum_gap", "band_order")
        .agg(
            pl.len().alias("artists")
        )
        .with_columns(
            (
                pl.col("artists") / pl.col("artists").sum() * 100
            )
            .round(1)
            .alias("artist_pct")
        )
        .sort("band_order")
        .drop("band_order")
        .collect()
    )

    return_gap_bands
    return


@app.cell
def _(artist_active_days, artist_history, pl):
    longest_return_examples = (
        artist_active_days
        .filter(pl.col("gap_days").is_not_null())
        .join(
            artist_history.select(
                "artist_name",
                "scrobbles",
                "active_days",
            ),
            on="artist_name",
            how="left",
        )
        .select(
            "artist_name",
            "previous_active_date",
            pl.col("date").alias("returned_on"),
            "gap_days",
            "scrobbles",
            "active_days",
        )
        .sort("gap_days", descending=True)
        .head(20)
        .collect()
    )

    longest_return_examples
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Long returns are also common enough to matter. Among artists heard on more than one day, **38.3%** have at least one gap of six months or more, and **22.4%** have a gap of at least a year. Some artists return after more than five years.

    This supports letting artists become relevant again after long quiet periods.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we learned

    - The listening history is continuous and the main fields are complete, but listening behaviour changes a lot over time.
    - Artist and track names have real identity inconsistencies, while MBIDs are useful but incomplete.
    - Most artists are sampled briefly, while some return after very long gaps.
    - A future multi-user version should adapt to each user's listening level and allow artists to re-enter after inactivity.

    Next, these findings can guide the cleaned data layer and the breakout-target analysis.
    """)
    return


if __name__ == "__main__":
    app.run()
