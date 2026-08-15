from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import polars as pl

from sonora.data.comparison_normalization import normalize_for_comparison

_REQUIRED_EVENT_COLUMNS = {"artist_name"}
_REQUIRED_ALIAS_COLUMNS = {"observed_artist_name", "artist_name_comparison"}
_BLOCK_STOPWORDS = frozenset({"and", "the"})
_MIN_NGRAM_SIZE = 2
_MIN_BLOCK_SIZE = 2


@dataclass(frozen=True, slots=True)
class ArtistCandidateConfig:
    """High-recall blocking settings for artist candidate generation."""

    ngram_size: int = 3
    min_shared_ngrams: int = 2
    min_token_length: int = 3
    max_block_size: int = 100

    def __post_init__(self) -> None:
        if self.ngram_size < _MIN_NGRAM_SIZE:
            raise ValueError("ngram_size must be at least 2")
        if self.min_shared_ngrams < 1:
            raise ValueError("min_shared_ngrams must be at least 1")
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be at least 1")
        if self.max_block_size < _MIN_BLOCK_SIZE:
            raise ValueError("max_block_size must be at least 2")


DEFAULT_ARTIST_CANDIDATE_CONFIG = ArtistCandidateConfig()


def collect_artist_aliases(events: pl.LazyFrame) -> pl.DataFrame:
    """Collect distinct observed artist names with derived comparison names."""
    _validate_event_schema(events)

    aliases = (
        events.select(pl.col("artist_name").alias("observed_artist_name"))
        .unique()
        .with_columns(
            normalize_for_comparison(pl.col("observed_artist_name")).alias(
                "artist_name_comparison"
            )
        )
        .sort("observed_artist_name")
        .collect()
    )
    _validate_alias_values(aliases)
    return aliases


def generate_artist_candidate_pairs(
    aliases: pl.DataFrame,
    *,
    config: ArtistCandidateConfig = DEFAULT_ARTIST_CANDIDATE_CONFIG,
) -> pl.DataFrame:
    """Generate plausible artist-alias pairs without all-vs-all comparison."""
    _validate_alias_schema(aliases)
    _validate_alias_values(aliases)

    aliases = aliases.select("observed_artist_name", "artist_name_comparison").sort(
        "observed_artist_name"
    )
    records = list(aliases.iter_rows(named=True))

    exact_blocks: dict[str, list[int]] = defaultdict(list)
    token_blocks: dict[str, list[int]] = defaultdict(list)
    ngram_blocks: dict[str, list[int]] = defaultdict(list)

    for index, record in enumerate(records):
        comparison_name = record["artist_name_comparison"]
        exact_blocks[comparison_name].append(index)

        for token in _significant_tokens(comparison_name, config=config):
            token_blocks[token].append(index)

        for ngram in _character_ngrams(comparison_name, size=config.ngram_size):
            ngram_blocks[ngram].append(index)

    exact_pairs: set[tuple[int, int]] = set()
    _add_block_pairs(exact_blocks.values(), exact_pairs)

    shared_token_counts: Counter[tuple[int, int]] = Counter()
    _count_capped_block_pairs(
        token_blocks.values(),
        shared_token_counts,
        max_block_size=config.max_block_size,
    )

    shared_ngram_counts: Counter[tuple[int, int]] = Counter()
    _count_capped_block_pairs(
        ngram_blocks.values(),
        shared_ngram_counts,
        max_block_size=config.max_block_size,
    )

    ngram_pairs = {
        pair
        for pair, shared_count in shared_ngram_counts.items()
        if shared_count >= config.min_shared_ngrams
    }
    candidate_pairs = exact_pairs | set(shared_token_counts) | ngram_pairs

    return _candidate_frame(
        records,
        candidate_pairs,
        exact_pairs=exact_pairs,
        shared_token_counts=shared_token_counts,
        shared_ngram_counts=shared_ngram_counts,
    )


def _validate_event_schema(events: pl.LazyFrame) -> None:
    columns = set(events.collect_schema().names())
    missing = sorted(_REQUIRED_EVENT_COLUMNS - columns)
    if missing:
        raise ValueError(f"Listening events are missing required columns: {missing}")


def _validate_alias_schema(aliases: pl.DataFrame) -> None:
    columns = set(aliases.columns)
    missing = sorted(_REQUIRED_ALIAS_COLUMNS - columns)
    if missing:
        raise ValueError(f"Artist aliases are missing required columns: {missing}")


def _validate_alias_values(aliases: pl.DataFrame) -> None:
    if aliases.height == 0:
        return

    invalid = aliases.select(
        pl.col(column).is_null().sum().alias(column)
        for column in _REQUIRED_ALIAS_COLUMNS
        if column in aliases.columns
    ).row(0, named=True)
    invalid = {column: count for column, count in invalid.items() if count}
    if invalid:
        details = ", ".join(f"{column}={count}" for column, count in invalid.items())
        raise ValueError(f"Artist aliases contain missing required values: {details}")

    if "observed_artist_name" in aliases.columns:
        duplicate_count = aliases.select(
            pl.col("observed_artist_name").is_duplicated().sum()
        ).item()
        if duplicate_count:
            raise ValueError("Artist aliases must contain unique observed artist names")


def _significant_tokens(
    comparison_name: str,
    *,
    config: ArtistCandidateConfig,
) -> set[str]:
    return {
        token
        for token in comparison_name.split()
        if len(token) >= config.min_token_length and token not in _BLOCK_STOPWORDS
    }


def _character_ngrams(value: str, *, size: int) -> set[str]:
    compact = value.replace(" ", "")
    padded = f"^{compact}$"
    if len(padded) <= size:
        return {padded}
    return {padded[index : index + size] for index in range(len(padded) - size + 1)}


def _add_block_pairs(
    blocks,
    candidate_pairs: set[tuple[int, int]],
) -> None:
    for block in blocks:
        unique_indices = sorted(set(block))
        if len(unique_indices) >= 2:
            candidate_pairs.update(combinations(unique_indices, 2))


def _count_capped_block_pairs(
    blocks,
    pair_counts: Counter[tuple[int, int]],
    *,
    max_block_size: int,
) -> None:
    for block in blocks:
        unique_indices = sorted(set(block))
        if 2 <= len(unique_indices) <= max_block_size:
            pair_counts.update(combinations(unique_indices, 2))


def _candidate_frame(
    records: list[dict[str, str]],
    candidate_pairs: set[tuple[int, int]],
    *,
    exact_pairs: set[tuple[int, int]],
    shared_token_counts: Counter[tuple[int, int]],
    shared_ngram_counts: Counter[tuple[int, int]],
) -> pl.DataFrame:
    ordered_pairs = sorted(
        candidate_pairs,
        key=lambda pair: (
            records[pair[0]]["observed_artist_name"],
            records[pair[1]]["observed_artist_name"],
        ),
    )

    return pl.DataFrame(
        {
            "observed_artist_name_left": pl.Series(
                [records[left]["observed_artist_name"] for left, _ in ordered_pairs],
                dtype=pl.String,
            ),
            "artist_name_comparison_left": pl.Series(
                [records[left]["artist_name_comparison"] for left, _ in ordered_pairs],
                dtype=pl.String,
            ),
            "observed_artist_name_right": pl.Series(
                [records[right]["observed_artist_name"] for _, right in ordered_pairs],
                dtype=pl.String,
            ),
            "artist_name_comparison_right": pl.Series(
                [
                    records[right]["artist_name_comparison"]
                    for _, right in ordered_pairs
                ],
                dtype=pl.String,
            ),
            "blocking_exact_name_match": pl.Series(
                [pair in exact_pairs for pair in ordered_pairs],
                dtype=pl.Boolean,
            ),
            "blocking_shared_token_count": pl.Series(
                [shared_token_counts[pair] for pair in ordered_pairs],
                dtype=pl.UInt32,
            ),
            "blocking_shared_ngram_count": pl.Series(
                [shared_ngram_counts[pair] for pair in ordered_pairs],
                dtype=pl.UInt32,
            ),
        }
    )
