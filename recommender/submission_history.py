import json
from collections import Counter
from pathlib import Path

USER_SUBMISSIONS_DIR = Path("data/raw/user_submissions")
ACCEPTED_VERDICT = "OK"


def get_user_submissions_path(handle, submissions_dir=USER_SUBMISSIONS_DIR):
    return Path(submissions_dir) / f"{handle}.json"


def load_user_submissions(handle, submissions_dir=USER_SUBMISSIONS_DIR):
    submissions_path = get_user_submissions_path(handle, submissions_dir)
    return load_user_submissions_from_path(submissions_path)


def load_user_submissions_from_path(submissions_path):
    submissions = json.loads(Path(submissions_path).read_text(encoding="utf-8"))

    if not isinstance(submissions, list):
        raise ValueError("User submissions JSON must contain a list")

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

    return {
        "problem_key": make_problem_key(ordered_submissions[0]),
        "problem": ordered_submissions[-1].get("problem", {}),
        "is_solved": bool(accepted_submissions),
        "submission_count": len(ordered_submissions),
        "failed_attempt_count": failed_attempt_count,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "first_submission_time": ordered_submissions[0].get("creationTimeSeconds"),
        "last_submission_time": ordered_submissions[-1].get("creationTimeSeconds"),
        "first_accepted_time": first_accepted_time,
    }


def analyze_submissions_by_problem(submissions):
    grouped_submissions = group_submissions_by_problem(submissions)
    return {
        problem_key: analyze_problem_submissions(problem_submissions)
        for problem_key, problem_submissions in grouped_submissions.items()
    }
