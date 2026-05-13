if __package__ is None or __package__ == "":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

import argparse
from pathlib import Path

import numpy as np

from recommender.submission_history import USER_DATA_DIR
from recommender.tag_idf import TAG_STATS_PATH, load_tag_idf_vector
from recommender.tag_tf import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MISSING_RATING_WEIGHT,
    DEFAULT_RATING_SCALE,
    build_user_tag_tf_components,
    calculate_tag_tf_ratio_vector,
)

DEFAULT_IDF_WEIGHT_CAP = 2.0
DEFAULT_CONFIDENCE_SMOOTHING = 0.1


def calculate_idf_weight_vector(
    tag_idf_vector,
    idf_weight_cap=DEFAULT_IDF_WEIGHT_CAP,
):
    if idf_weight_cap <= 0:
        raise ValueError("idf_weight_cap must be positive")

    return np.minimum(np.sqrt(tag_idf_vector), idf_weight_cap)


def calculate_confidence_vector(
    exposure_vector,
    confidence_smoothing=DEFAULT_CONFIDENCE_SMOOTHING,
):
    if confidence_smoothing < 0:
        raise ValueError("confidence_smoothing must not be negative")

    exposure_vector = np.asarray(exposure_vector, dtype=np.float64)
    denominator = exposure_vector + confidence_smoothing
    return np.divide(
        exposure_vector,
        denominator,
        out=np.zeros_like(exposure_vector),
        where=denominator > 0,
    )


def build_user_tag_tfidf_vector(
    handle,
    users_dir=USER_DATA_DIR,
    tag_stats_path=TAG_STATS_PATH,
    user_rating=None,
    current_time_seconds=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    rating_scale=DEFAULT_RATING_SCALE,
    missing_rating_weight=DEFAULT_MISSING_RATING_WEIGHT,
    idf_weight_cap=DEFAULT_IDF_WEIGHT_CAP,
    confidence_smoothing=DEFAULT_CONFIDENCE_SMOOTHING,
    normalize_tf=True,
):
    (
        tag_names,
        tag_tf_vector,
        tag_idf_vector,
        _,
        _,
        tag_score_vector,
    ) = build_user_tag_tfidf_components(
        handle,
        users_dir=users_dir,
        tag_stats_path=tag_stats_path,
        user_rating=user_rating,
        current_time_seconds=current_time_seconds,
        half_life_days=half_life_days,
        rating_scale=rating_scale,
        missing_rating_weight=missing_rating_weight,
        idf_weight_cap=idf_weight_cap,
        confidence_smoothing=confidence_smoothing,
        normalize_tf=normalize_tf,
    )
    return tag_names, tag_tf_vector, tag_idf_vector, tag_score_vector


def build_user_tag_tfidf_components(
    handle,
    users_dir=USER_DATA_DIR,
    tag_stats_path=TAG_STATS_PATH,
    user_rating=None,
    current_time_seconds=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    rating_scale=DEFAULT_RATING_SCALE,
    missing_rating_weight=DEFAULT_MISSING_RATING_WEIGHT,
    idf_weight_cap=DEFAULT_IDF_WEIGHT_CAP,
    confidence_smoothing=DEFAULT_CONFIDENCE_SMOOTHING,
    normalize_tf=True,
):
    _, tag_names, _, tag_idf_vector = load_tag_idf_vector(tag_stats_path)
    weakness_sum_vector, exposure_vector = build_user_tag_tf_components(
        handle,
        tag_names,
        users_dir=users_dir,
        user_rating=user_rating,
        current_time_seconds=current_time_seconds,
        half_life_days=half_life_days,
        rating_scale=rating_scale,
        missing_rating_weight=missing_rating_weight,
    )

    if normalize_tf:
        tag_tf_vector = calculate_tag_tf_ratio_vector(
            weakness_sum_vector,
            exposure_vector,
        )
    else:
        tag_tf_vector = weakness_sum_vector

    idf_weight_vector = calculate_idf_weight_vector(
        tag_idf_vector,
        idf_weight_cap=idf_weight_cap,
    )
    confidence_vector = calculate_confidence_vector(
        exposure_vector,
        confidence_smoothing=confidence_smoothing,
    )
    tag_score_vector = tag_tf_vector * idf_weight_vector * confidence_vector

    return (
        tag_names,
        tag_tf_vector,
        tag_idf_vector,
        idf_weight_vector,
        confidence_vector,
        tag_score_vector,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("handle")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--users-dir", type=Path, default=USER_DATA_DIR)
    parser.add_argument("--tag-stats-path", type=Path, default=TAG_STATS_PATH)
    parser.add_argument("--user-rating", type=int)
    parser.add_argument("--half-life-days", type=float, default=DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--rating-scale", type=float, default=DEFAULT_RATING_SCALE)
    parser.add_argument("--idf-weight-cap", type=float, default=DEFAULT_IDF_WEIGHT_CAP)
    parser.add_argument(
        "--confidence-smoothing",
        type=float,
        default=DEFAULT_CONFIDENCE_SMOOTHING,
    )
    parser.add_argument(
        "--missing-rating-weight",
        type=float,
        default=DEFAULT_MISSING_RATING_WEIGHT,
    )
    args = parser.parse_args()

    (
        tag_names,
        tag_tf_vector,
        tag_idf_vector,
        idf_weight_vector,
        confidence_vector,
        tag_score_vector,
    ) = (
        build_user_tag_tfidf_components(
            args.handle,
            users_dir=args.users_dir,
            tag_stats_path=args.tag_stats_path,
            user_rating=args.user_rating,
            half_life_days=args.half_life_days,
            rating_scale=args.rating_scale,
            missing_rating_weight=args.missing_rating_weight,
            idf_weight_cap=args.idf_weight_cap,
            confidence_smoothing=args.confidence_smoothing,
        )
    )

    if args.all:
        indices = range(len(tag_names))
    else:
        indices = np.argsort(tag_score_vector)[::-1][:args.top]

    for index in indices:
        print(
            f"{tag_names[index]}: "
            f"tf={tag_tf_vector[index]:.6f}, "
            f"idf={tag_idf_vector[index]:.6f}, "
            f"idf_weight={idf_weight_vector[index]:.6f}, "
            f"confidence={confidence_vector[index]:.6f}, "
            f"score={tag_score_vector[index]:.6f}"
        )


if __name__ == "__main__":
    main()
