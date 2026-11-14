import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from time import perf_counter

    import marimo as mo
    import numpy as np
    import polars as pl

    from sonora.data.paths import DEFAULT_DATA_PATHS
    from sonora.similarity.descriptions import clean_artist_description

    paths = DEFAULT_DATA_PATHS
    return clean_artist_description, mo, np, paths, perf_counter, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Can BGE-M3 process Sonora's artist descriptions?

    Check input lengths and inference time for cleaned summaries and biographies.
    """)
    return


@app.cell
def _(clean_artist_description, paths, pl):
    descriptions = (
        pl.read_parquet(paths.curated_artist_descriptions)
        .filter(pl.col("status") == "success")
        .with_columns(
            pl.col("summary")
            .map_elements(clean_artist_description, return_dtype=pl.String)
            .alias("summary_text"),
            pl.col("content")
            .map_elements(clean_artist_description, return_dtype=pl.String)
            .alias("content_text"),
        )
        .filter(
            pl.col("summary_text").is_not_null()
            & pl.col("content_text").is_not_null()
        )
        .with_columns(
            pl.col("content_text").str.count_matches(r"\b\w+\b").alias("words")
        )
        .sort("artist_id")
    )
    if descriptions["artist_id"].n_unique() != descriptions.height:
        raise ValueError("Descriptions contain duplicate artist IDs")
    return (descriptions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which descriptions should we test?

    A fixed sample of short, medium and longer biographies.
    """)
    return


@app.cell
def _(descriptions, pl):
    _groups = (
        descriptions.filter(pl.col("words") < 150),
        descriptions.filter((pl.col("words") >= 150) & (pl.col("words") < 500)),
        descriptions.filter((pl.col("words") >= 500) & (pl.col("words") <= 1200)),
    )
    pilot = pl.concat(
        [group.sample(n=min(4, group.height), seed=42) for group in _groups]
    ).sort("artist_id")
    if pilot.is_empty():
        raise ValueError("No artist descriptions available for the pilot")
    pilot.select("canonical_name", "words").sort("words")
    return (pilot,)


@app.cell
def _(mo):
    run_pilot = mo.ui.run_button(label="Run BGE-M3 pilot")
    run_pilot
    return (run_pilot,)


@app.cell
def _(descriptions, mo, np, perf_counter, pilot, pl, run_pilot):
    mo.stop(not run_pilot.value)

    from sentence_transformers import SentenceTransformer

    _start = perf_counter()
    _model = SentenceTransformer("BAAI/bge-m3")
    _load_seconds = perf_counter() - _start
    _max_tokens = _model.max_seq_length

    def _token_lengths(texts):
        sizes = []
        for _offset in range(0, len(texts), 32):
            _batch = texts[_offset : _offset + 32]
            sizes.extend(
                len(tokens)
                for tokens in _model.tokenizer(
                    _batch, add_special_tokens=True, truncation=False
                )["input_ids"]
            )
        return sizes

    _all_lengths = _token_lengths(descriptions["content_text"].to_list())
    _pilot_summaries = pilot["summary_text"].to_list()
    _pilot_biographies = pilot["content_text"].to_list()
    _pilot_summary_lengths = _token_lengths(_pilot_summaries)
    _pilot_bio_lengths = _token_lengths(_pilot_biographies)
    if any(length > _max_tokens for length in _pilot_bio_lengths):
        raise ValueError("The pilot contains biographies longer than the model limit")

    _model.encode(
        _pilot_summaries[:1],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _times = {}
    for _label, _texts in (
        ("Summaries", _pilot_summaries),
        ("Full biographies", _pilot_biographies),
    ):
        _start = perf_counter()
        _embeddings = _model.encode(
            _texts,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        _times[_label] = perf_counter() - _start
        if _embeddings.shape != (len(_texts), 1024) or not np.isfinite(
            _embeddings
        ).all():
            raise ValueError("Unexpected BGE-M3 embeddings")

    _limit_count = sum(length > _max_tokens for length in _all_lengths)
    _sample_lengths = pl.DataFrame(
        {
            "Artist": pilot["canonical_name"].to_list(),
            "Summary tokens": _pilot_summary_lengths,
            "Biography tokens": _pilot_bio_lengths,
        }
    ).sort("Biography tokens")
    _timings = pl.DataFrame(
        {
            "Input": ["Summaries", "Full biographies"],
            "Artists": [pilot.height, pilot.height],
            "Seconds": [
                round(_times["Summaries"], 1),
                round(_times["Full biographies"], 1),
            ],
            "Seconds/artist": [
                round(_times["Summaries"] / pilot.height, 2),
                round(_times["Full biographies"] / pilot.height, 2),
            ],
        }
    )
    mo.vstack(
        [
            mo.md(
                f"**Model load:** {_load_seconds:.1f}s | **Device:** {_model.device} | "
                f"**Maximum input:** {_max_tokens:,} tokens"
            ),
            mo.md(
                f"**{_limit_count:,} of {descriptions.height:,} biographies** "
                "exceed the model's input limit."
            ),
            mo.ui.table(_timings, selection=None, page_size=2),
            mo.ui.table(_sample_lengths, selection=None, page_size=pilot.height),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
