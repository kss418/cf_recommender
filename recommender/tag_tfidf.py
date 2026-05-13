if __package__ is None or __package__ == "":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

import argparse

import numpy as np

from recommender.tag_idf import load_tag_idf_vector
from recommender.tag_tf import (
    build_user_tag_tf_components,
    calculate_tag_tf_ratio_vector,
)

DEFAULT_IDF_WEIGHT_CAP = 2.0
DEFAULT_CONFIDENCE_SMOOTHING = 0.1


def calculate_idf_weight_vector(tag_idf_vector):
    if DEFAULT_IDF_WEIGHT_CAP <= 0:
        raise ValueError("idf_weight_cap must be positive")

    return np.minimum(np.sqrt(tag_idf_vector), DEFAULT_IDF_WEIGHT_CAP)


def calculate_confidence_vector(exposure_vector):
    if DEFAULT_CONFIDENCE_SMOOTHING < 0:
        raise ValueError("confidence_smoothing must not be negative")

    exposure_vector = np.asarray(exposure_vector, dtype=np.float64)
    denominator = exposure_vector + DEFAULT_CONFIDENCE_SMOOTHING
    return np.divide(
        exposure_vector,
        denominator,
        out=np.zeros_like(exposure_vector),
        where=denominator > 0,
    )


def build_user_tag_tfidf_vector(handle):
    (
        tag_names,
        tag_tf_vector,
        tag_idf_vector,
        _,
        _,
        tag_score_vector,
    ) = build_user_tag_tfidf_components(
        handle,
    )
    return tag_names, tag_tf_vector, tag_idf_vector, tag_score_vector


def build_user_tag_tfidf_components(handle):
    _, tag_names, _, tag_idf_vector = load_tag_idf_vector()
    weakness_sum_vector, exposure_vector = build_user_tag_tf_components(
        handle,
        tag_names,
    )

    tag_tf_vector = calculate_tag_tf_ratio_vector(
        weakness_sum_vector,
        exposure_vector,
    )
    idf_weight_vector = calculate_idf_weight_vector(tag_idf_vector)
    confidence_vector = calculate_confidence_vector(exposure_vector)
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
