from collections import Counter
from pathlib import Path

from data_pipeline.data_loader import (
    get_data_path,
    get_user_data_path as get_config_user_data_path,
    load_json,
)

USER_DATA_DIR = get_data_path("users_dir")
ACCEPTED_VERDICT = "OK"


def get_user_data_path(handle, users_dir=USER_DATA_DIR):
    if users_dir == USER_DATA_DIR:
        return get_config_user_data_path(handle)
    return Path(users_dir) / f"{handle}.json"


def load_user_data(handle, users_dir=USER_DATA_DIR):
    user_data_path = get_user_data_path(handle, users_dir)
    return load_user_data_from_path(user_data_path)


def load_user_data_from_path(user_data_path):
    user_data = load_json(user_data_path)

    if not isinstance(user_data, dict):
        raise ValueError("User data JSON must contain an object")

    return user_data


def load_user_submissions(handle, users_dir=USER_DATA_DIR):
    user_data = load_user_data(handle, users_dir)
    return extract_user_submissions(user_data)


def load_user_submissions_from_path(user_data_path):
    user_data = load_user_data_from_path(user_data_path)
    return extract_user_submissions(user_data)


def load_user_info(handle, users_dir=USER_DATA_DIR):
    return load_user_data(handle, users_dir).get("info", {})


def load_user_rating_history(handle, users_dir=USER_DATA_DIR):
    return load_user_data(handle, users_dir).get("rating_history", [])


def extract_user_submissions(user_data):
    submissions = user_data.get("submissions")

    if not isinstance(submissions, list):
        raise ValueError("User data JSON must contain a submissions list")

    return submissions


def make_problem_key(submission):
    problem = submission.get("problem", {})
    contest_id = problem.get("contestId", submission.get("contestId"))
    problem_index = problem.get("index")

    if contest_id is None or problem_index is None:
        raise ValueError("Submission is missing problem contestId or index")

    return contest_id, problem_index


def group_submissions_by_problem(submissions):
    grouped_submissions = {}
    for submission in submissions:
        problem_key = make_problem_key(submission)
        grouped_submissions.setdefault(problem_key, []).append(submission)
    return grouped_submissions


def analyze_problem_submissions(submissions):
    if not submissions:
        raise ValueError("Problem submissions must not be empty")

    ordered_submissions = sorted(
        submissions,
        key=lambda submission: submission.get("creationTimeSeconds", 0),
    )
    accepted_submissions = [
        submission
        for submission in ordered_submissions
        if submission.get("verdict") == ACCEPTED_VERDICT
    ]
    first_accepted_time = (
        accepted_submissions[0].get("creationTimeSeconds")
        if accepted_submissions
        else None
    )

    failed_attempt_count = sum(
        1
        for submission in ordered_submissions
        if submission.get("verdict") != ACCEPTED_VERDICT
        and (
            first_accepted_time is None
            or submission.get("creationTimeSeconds", 0) < first_accepted_time
        )
    )
    verdict_counts = Counter(
        submission.get("verdict", "UNKNOWN")
        for submission in ordered_submissions
    )
    is_solved = bool(accepted_submissions)
    last_submission_time = ordered_submissions[-1].get("creationTimeSeconds")
    decay_reference_time = (
        first_accepted_time
        if is_solved
        else last_submission_time
    )

    return {
        "problem_key": make_problem_key(ordered_submissions[0]),
        "problem": ordered_submissions[-1].get("problem", {}),
        "is_solved": is_solved,
        "submission_count": len(ordered_submissions),
        "failed_attempt_count": failed_attempt_count,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "first_submission_time": ordered_submissions[0].get("creationTimeSeconds"),
        "last_submission_time": last_submission_time,
        "first_accepted_time": first_accepted_time,
        "decay_reference_time": decay_reference_time,
    }


def analyze_submissions_by_problem(submissions):
    grouped_submissions = group_submissions_by_problem(submissions)
    return {
        problem_key: analyze_problem_submissions(problem_submissions)
        for problem_key, problem_submissions in grouped_submissions.items()
    }
