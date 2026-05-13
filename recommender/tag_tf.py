import time

import numpy as np

from recommender.submission_history import (
    analyze_submissions_by_problem,
    load_user_info,
    load_user_submissions,
)

SECONDS_PER_DAY = 24 * 60 * 60
DEFAULT_HALF_LIFE_DAYS = 60
DEFAULT_RATING_SCALE = 400
DEFAULT_MISSING_RATING_WEIGHT = 0.5
SOLVED_WITHOUT_FAILURE_SIGNAL = 0.0
SOLVED_WITH_FEW_FAILURES_SIGNAL = 0.3
SOLVED_WITH_MANY_FAILURES_SIGNAL = 0.6
UNSOLVED_SIGNAL = 1.0
FEW_FAILURES_THRESHOLD = 2


def sort_submissions_by_time(submissions):
    return sorted(
        submissions,
        key=lambda submission: submission.get("creationTimeSeconds", 0),
        reverse=True,
    )


def extract_submission_times(submissions):
    return np.array(
        [
            submission["creationTimeSeconds"]
            for submission in submissions
        ],
        dtype=np.float64,
    )


def calculate_exponential_decay(age_days):
    if DEFAULT_HALF_LIFE_DAYS <= 0:
        raise ValueError("half_life_days must be positive")

    lambda_ = np.log(2) / DEFAULT_HALF_LIFE_DAYS
    return np.exp(-lambda_ * age_days)


def build_decay_weight_vector(submission_times):
    submission_times = np.asarray(submission_times, dtype=np.float64)

    if submission_times.size == 0:
        raise ValueError("submission_times must not be empty")

    current_time_seconds = time.time()
    age_days = (current_time_seconds - submission_times) / SECONDS_PER_DAY
    age_days = np.maximum(age_days, 0)
    return calculate_exponential_decay(age_days)


def extract_problem_decay_reference_times(problem_analysis_by_key):
    reference_times = []
    for analysis in problem_analysis_by_key.values():
        reference_time = analysis.get("decay_reference_time")
        if reference_time is None:
            raise ValueError("Problem analysis is missing decay_reference_time")
        reference_times.append(reference_time)

    return np.array(reference_times, dtype=np.float64)


def build_problem_decay_weight_vector(problem_analysis_by_key):
    problem_keys = list(problem_analysis_by_key)
    reference_times = extract_problem_decay_reference_times(problem_analysis_by_key)
    decay_weight_vector = build_decay_weight_vector(reference_times)
    return problem_keys, decay_weight_vector


def calculate_difficulty_weight(
    user_rating,
    problem_rating,
):
    if DEFAULT_RATING_SCALE <= 0:
        raise ValueError("rating_scale must be positive")

    if user_rating is None or problem_rating is None:
        return DEFAULT_MISSING_RATING_WEIGHT

    rating_gap = (problem_rating - user_rating) / DEFAULT_RATING_SCALE
    return 1 / (1 + np.power(10, rating_gap))


def build_problem_difficulty_weight_vector(
    problem_analysis_by_key,
    user_rating,
):
    problem_keys = list(problem_analysis_by_key)
    difficulty_weight_vector = np.array(
        [
            calculate_difficulty_weight(
                user_rating,
                problem_analysis_by_key[problem_key]
                .get("problem", {})
                .get("rating"),
            )
            for problem_key in problem_keys
        ],
        dtype=np.float64,
    )
    return problem_keys, difficulty_weight_vector


def calculate_weakness_signal(problem_analysis):
    if not problem_analysis.get("is_solved"):
        return UNSOLVED_SIGNAL

    failed_attempt_count = problem_analysis.get("failed_attempt_count", 0)
    if failed_attempt_count == 0:
        return SOLVED_WITHOUT_FAILURE_SIGNAL

    if failed_attempt_count <= FEW_FAILURES_THRESHOLD:
        return SOLVED_WITH_FEW_FAILURES_SIGNAL

    return SOLVED_WITH_MANY_FAILURES_SIGNAL


def build_tag_index(tag_names):
    return {
        tag: index
        for index, tag in enumerate(tag_names)
    }


def build_tag_tf_vector(
    problem_analysis_by_key,
    tag_names,
    user_rating,
):
    weakness_sum_vector, exposure_vector = build_tag_tf_components(
        problem_analysis_by_key,
        tag_names,
        user_rating=user_rating,
    )

    return calculate_tag_tf_ratio_vector(weakness_sum_vector, exposure_vector)


def calculate_tag_tf_ratio_vector(weakness_sum_vector, exposure_vector):
    weakness_sum_vector = np.asarray(weakness_sum_vector, dtype=np.float64)
    exposure_vector = np.asarray(exposure_vector, dtype=np.float64)
    return np.divide(
        weakness_sum_vector,
        exposure_vector,
        out=np.zeros_like(weakness_sum_vector),
        where=exposure_vector > 0,
    )


def build_tag_tf_components(
    problem_analysis_by_key,
    tag_names,
    user_rating,
):
    tag_index = build_tag_index(tag_names)
    weakness_sum_vector = np.zeros(len(tag_names), dtype=np.float64)
    exposure_vector = np.zeros(len(tag_names), dtype=np.float64)
    problem_keys, decay_weight_vector = build_problem_decay_weight_vector(
        problem_analysis_by_key
    )
    difficulty_keys, difficulty_weight_vector = build_problem_difficulty_weight_vector(
        problem_analysis_by_key,
        user_rating,
    )
    if problem_keys != difficulty_keys:
        raise ValueError("Problem weight vectors are not aligned")

    for problem_key, decay_weight, difficulty_weight in zip(
        problem_keys,
        decay_weight_vector,
        difficulty_weight_vector,
    ):
        problem_analysis = problem_analysis_by_key[problem_key]
        weakness_signal = calculate_weakness_signal(problem_analysis)
        problem_weight = decay_weight * difficulty_weight

        problem_tags = set(problem_analysis.get("problem", {}).get("tags", []))
        for tag in problem_tags:
            index = tag_index.get(tag)
            if index is None:
                continue

            exposure_vector[index] += problem_weight
            weakness_sum_vector[index] += problem_weight * weakness_signal

    return weakness_sum_vector, exposure_vector


def analyze_user_submissions_by_problem(handle):
    submissions = load_user_submissions(handle)
    submissions = sort_submissions_by_time(submissions)
    return analyze_submissions_by_problem(submissions)


def build_user_tag_tf_vector(
    handle,
    tag_names,
):
    user_rating = load_user_info(handle).get("rating")

    problem_analysis_by_key = analyze_user_submissions_by_problem(handle)
    return build_tag_tf_vector(
        problem_analysis_by_key,
        tag_names,
        user_rating=user_rating,
    )


def build_user_tag_tf_components(
    handle,
    tag_names,
):
    user_rating = load_user_info(handle).get("rating")

    problem_analysis_by_key = analyze_user_submissions_by_problem(handle)
    return build_tag_tf_components(
        problem_analysis_by_key,
        tag_names,
        user_rating=user_rating,
    )
