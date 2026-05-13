from data_pipeline.data_loader import get_data_path, load_json
from recommender.submission_history import (
    ACCEPTED_VERDICT,
    load_user_info,
    load_user_submissions,
    make_problem_key,
)
from recommender.tag_tfidf import build_user_tag_tfidf_components

CODEFORCES_PROBLEMS_PATH = get_data_path("codeforces_problems")
DEFAULT_MIN_RATING_OFFSET = -400
DEFAULT_MAX_RATING_OFFSET = 200
DEFAULT_RECOMMENDATION_LIMIT = 20
DEFAULT_PRIMARY_TAG_QUOTA = 2


def load_problemset():
    problemset = load_json(CODEFORCES_PROBLEMS_PATH)
    problems = problemset.get("problems")

    if not isinstance(problems, list):
        raise ValueError("Problemset JSON must contain a problems list")

    return problems


def load_user_submission_history(handle):
    return load_user_submissions(handle)


def make_problem_key_from_problem(problem):
    contest_id = problem.get("contestId")
    problem_index = problem.get("index")

    if contest_id is None or problem_index is None:
        raise ValueError("Problem is missing contestId or index")

    return contest_id, problem_index


def collect_accepted_problem_keys(submissions):
    accepted_problem_keys = set()
    for submission in submissions:
        if submission.get("verdict") != ACCEPTED_VERDICT:
            continue

        accepted_problem_keys.add(make_problem_key(submission))

    return accepted_problem_keys


def filter_accepted_problems(problems, accepted_problem_keys):
    unsolved_problems = []
    for problem in problems:
        problem_key = make_problem_key_from_problem(problem)
        if problem_key in accepted_problem_keys:
            continue

        unsolved_problems.append(problem)

    return unsolved_problems


def filter_problems_by_rating_range(
    problems,
    user_rating,
):
    if user_rating is None:
        raise ValueError("User rating is required for rating range filtering")

    min_rating = user_rating + DEFAULT_MIN_RATING_OFFSET
    max_rating = user_rating + DEFAULT_MAX_RATING_OFFSET

    return [
        problem
        for problem in problems
        if problem.get("rating") is not None
        and min_rating <= problem["rating"] <= max_rating
    ]


def load_tag_tfidf_score_map(handle):
    (
        tag_names,
        _,
        _,
        _,
        _,
        tag_score_vector,
    ) = build_user_tag_tfidf_components(handle)

    return {
        tag: float(score)
        for tag, score in zip(tag_names, tag_score_vector)
    }


def calculate_problem_tfidf_score(
    problem,
    tag_tfidf_score_map,
):
    tag_scores = [
        tag_tfidf_score_map[tag]
        for tag in problem.get("tags", [])
        if tag in tag_tfidf_score_map
    ]

    if not tag_scores:
        return 0.0

    return sum(tag_scores) / len(tag_scores)


def calculate_problem_primary_tag(
    problem,
    tag_tfidf_score_map,
):
    tag_scores = [
        (tag, tag_tfidf_score_map[tag])
        for tag in problem.get("tags", [])
        if tag in tag_tfidf_score_map
    ]

    if not tag_scores:
        return None

    return max(tag_scores, key=lambda tag_score: tag_score[1])[0]


def rank_problems_by_tfidf(
    problems,
    tag_tfidf_score_map,
):
    scored_problems = []
    for problem in problems:
        scored_problem = dict(problem)
        scored_problem["tfidf_score"] = calculate_problem_tfidf_score(
            problem,
            tag_tfidf_score_map,
        )
        scored_problem["primary_tag"] = calculate_problem_primary_tag(
            problem,
            tag_tfidf_score_map,
        )
        scored_problems.append(scored_problem)

    return sorted(
        scored_problems,
        key=lambda problem: problem["tfidf_score"],
        reverse=True,
    )


def select_diverse_problems(ranked_problems):
    if DEFAULT_RECOMMENDATION_LIMIT <= 0:
        raise ValueError("limit must be positive")

    if DEFAULT_PRIMARY_TAG_QUOTA <= 0:
        raise ValueError("primary_tag_quota must be positive")

    selected_problems = []
    primary_tag_counts = {}

    for problem in ranked_problems:
        primary_tag = problem.get("primary_tag")
        selected_count = primary_tag_counts.get(primary_tag, 0)
        if selected_count >= DEFAULT_PRIMARY_TAG_QUOTA:
            continue

        selected_problem = dict(problem)
        selected_problem["recommendation_rank"] = len(selected_problems) + 1
        selected_problems.append(selected_problem)
        primary_tag_counts[primary_tag] = selected_count + 1

        if len(selected_problems) >= DEFAULT_RECOMMENDATION_LIMIT:
            break

    return selected_problems


def load_unsolved_problem_candidates(handle):
    submissions = load_user_submission_history(handle)
    user_info = load_user_info(handle)
    user_rating = user_info.get("rating")
    accepted_problem_keys = collect_accepted_problem_keys(submissions)
    problems = load_problemset()
    unsolved_problems = filter_accepted_problems(problems, accepted_problem_keys)
    candidate_problems = filter_problems_by_rating_range(
        unsolved_problems,
        user_rating,
    )
    tag_tfidf_score_map = load_tag_tfidf_score_map(
        handle,
    )
    ranked_candidate_problems = rank_problems_by_tfidf(
        candidate_problems,
        tag_tfidf_score_map,
    )
    recommended_problems = select_diverse_problems(ranked_candidate_problems)

    return {
        "user_rating": user_rating,
        "tag_tfidf_score_map": tag_tfidf_score_map,
        "submissions": submissions,
        "accepted_problem_keys": accepted_problem_keys,
        "problems": problems,
        "unsolved_problems": unsolved_problems,
        "candidate_problems": candidate_problems,
        "ranked_candidate_problems": ranked_candidate_problems,
        "recommended_problems": recommended_problems,
    }
