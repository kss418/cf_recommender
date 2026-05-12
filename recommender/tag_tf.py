import time

import numpy as np

from recommender.submission_history import (
    USER_DATA_DIR,
    analyze_submissions_by_problem,
    load_user_submissions,
)

SECONDS_PER_DAY = 24 * 60 * 60
DEFAULT_HALF_LIFE_DAYS = 60


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


def analyze_user_submissions_by_problem(
    handle,
    users_dir=USER_DATA_DIR,
):
    submissions = load_user_submissions(handle, users_dir)
    submissions = sort_submissions_by_time(submissions)
    return analyze_submissions_by_problem(submissions)
