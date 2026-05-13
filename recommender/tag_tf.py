import time

import numpy as np

from recommender.submission_history import (
    USER_DATA_DIR,
    analyze_submissions_by_problem,
    load_user_submissions,
)

SECONDS_PER_DAY = 24 * 60 * 60
DEFAULT_HALF_LIFE_DAYS = 60
SOLVED_WITHOUT_FAILURE_SIGNAL = 0.0
SOLVED_WITH_FEW_FAILURES_SIGNAL = 0.3
SOLVED_WITH_MANY_FAILURES_SIGNAL = 0.6
UNSOLVED_SIGNAL = 1.0
FEW_FAILURES_THRESHOLD = 2


def sort_submissions_by_time(submissions, newest_first=True):
    return sorted(
        submissions,
        key=lambda submission: submission.get("creationTimeSeconds", 0),
        reverse=newest_first,
    )


def extract_submission_times(submissions):
    return np.array(
        [
            submission["creationTimeSeconds"]
            for submission in submissions
        ],
        dtype=np.float64,
    )


def calculate_exponential_decay(age_days, half_life_days=DEFAULT_HALF_LIFE_DAYS):
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    lambda_ = np.log(2) / half_life_days
    return np.exp(-lambda_ * age_days)


def build_decay_weight_vector(
    submission_times,
    current_time_seconds=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
):
    submission_times = np.asarray(submission_times, dtype=np.float64)

    if submission_times.size == 0:
        raise ValueError("submission_times must not be empty")

    if current_time_seconds is None:
        current_time_seconds = time.time()

    age_days = (current_time_seconds - submission_times) / SECONDS_PER_DAY
    age_days = np.maximum(age_days, 0)
    return calculate_exponential_decay(age_days, half_life_days)


def extract_problem_decay_reference_times(problem_analysis_by_key):
    reference_times = []
    for analysis in problem_analysis_by_key.values():
        reference_time = analysis.get("decay_reference_time")
        if reference_time is None:
            raise ValueError("Problem analysis is missing decay_reference_time")
        reference_times.append(reference_time)

    return np.array(reference_times, dtype=np.float64)


def build_problem_decay_weight_vector(
    problem_analysis_by_key,
    current_time_seconds=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
):
    problem_keys = list(problem_analysis_by_key)
    reference_times = extract_problem_decay_reference_times(problem_analysis_by_key)
    decay_weight_vector = build_decay_weight_vector(
        reference_times,
        current_time_seconds=current_time_seconds,
        half_life_days=half_life_days,
    )
    return problem_keys, decay_weight_vector


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
    current_time_seconds=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    normalize=True,
):
    tag_index = build_tag_index(tag_names)
    weakness_sum_vector = np.zeros(len(tag_names), dtype=np.float64)
    exposure_vector = np.zeros(len(tag_names), dtype=np.float64)
    problem_keys, decay_weight_vector = build_problem_decay_weight_vector(
        problem_analysis_by_key,
        current_time_seconds=current_time_seconds,
        half_life_days=half_life_days,
    )

    for problem_key, decay_weight in zip(problem_keys, decay_weight_vector):
        problem_analysis = problem_analysis_by_key[problem_key]
        weakness_signal = calculate_weakness_signal(problem_analysis)

        problem_tags = set(problem_analysis.get("problem", {}).get("tags", []))
        for tag in problem_tags:
            index = tag_index.get(tag)
            if index is None:
                continue

            exposure_vector[index] += decay_weight
            weakness_sum_vector[index] += decay_weight * weakness_signal

    if not normalize:
        return weakness_sum_vector

    return np.divide(
        weakness_sum_vector,
        exposure_vector,
        out=np.zeros_like(weakness_sum_vector),
        where=exposure_vector > 0,
    )


def analyze_user_submissions_by_problem(
    handle,
    users_dir=USER_DATA_DIR,
):
    submissions = load_user_submissions(handle, users_dir)
    submissions = sort_submissions_by_time(submissions)
    return analyze_submissions_by_problem(submissions)


def build_user_tag_tf_vector(
    handle,
    tag_names,
    users_dir=USER_DATA_DIR,
    current_time_seconds=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    normalize=True,
):
    problem_analysis_by_key = analyze_user_submissions_by_problem(handle, users_dir)
    return build_tag_tf_vector(
        problem_analysis_by_key,
        tag_names,
        current_time_seconds=current_time_seconds,
        half_life_days=half_life_days,
        normalize=normalize,
    )
