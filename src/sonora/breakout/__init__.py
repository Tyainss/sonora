"""Daily first-breakout targets from curated listening history."""

from sonora.breakout.baselines import (
    score_listening_share_momentum,
    score_recent_active_days,
    score_recent_listening_share,
)
from sonora.breakout.events import detect_first_breakouts
from sonora.breakout.features import build_breakout_features
from sonora.breakout.history import aggregate_daily_listening, historical_user_bounds
from sonora.breakout.targets import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_RECENCY_DAYS,
    build_listening_gaps,
    eligible_segments,
    make_daily_targets,
)

__all__ = [
    "DEFAULT_HORIZON_DAYS",
    "DEFAULT_RECENCY_DAYS",
    "aggregate_daily_listening",
    "build_breakout_features",
    "build_listening_gaps",
    "detect_first_breakouts",
    "eligible_segments",
    "historical_user_bounds",
    "make_daily_targets",
    "score_listening_share_momentum",
    "score_recent_active_days",
    "score_recent_listening_share",
]
