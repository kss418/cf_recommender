from data_pipeline.data_loader import get_data_path, load_json
from recommender.submission_history import (
    ACCEPTED_VERDICT,
    USER_DATA_DIR,
    load_user_info,
    load_user_submissions,
    make_problem_key,
)
from recommender.tag_idf import TAG_STATS_PATH
from recommender.tag_tf import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MISSING_RATING_WEIGHT,
    DEFAULT_RATING_SCALE,
)
from recommender.tag_tfidf import (
    DEFAULT_CONFIDENCE_SMOOTHING,
    DEFAULT_IDF_WEIGHT_CAP,
    build_user_tag_tfidf_components,
)

CODEFORCES_PROBLEMS_PATH = get_data_path("codeforces_problems")
DEFAULT_MIN_RATING_OFFSET = -400
DEFAULT_MAX_RATING_OFFSET = 200
DEFAULT_RECOMMENDATION_LIMIT = 20
DEFAULT_PRIMARY_TAG_QUOTA = 2


def load_problemset(problemset_path=CODEFORCES_PROBLEMS_PATH):
    problemset = load_json(problemset_path)
    problems = problemset.get("problems")

    if not isinstance(problems, list):
        raise ValueError("Problemset JSON must contain a problems list")

    return problems


def load_user_submission_history(handle, users_dir=USER_DATA_DIR):
    return load_user_submissions(handle, users_dir)


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
    min_rating_offset=DEFAULT_MIN_RATING_OFFSET,
    max_rating_offset=DEFAULT_MAX_RATING_OFFSET,
):
    if user_rating is None:
        raise ValueError("User rating is required for rating range filtering")

    min_rating = user_rating + min_rating_offset
    max_rating = user_rating + max_rating_offset

    return [
        problem
        for problem in problems
        if problem.get("rating") is not None
        and min_rating <= problem["rating"] <= max_rating
    ]


def load_tag_tfidf_score_map(
    handle,
    users_dir=USER_DATA_DIR,
    tag_stats_path=TAG_STATS_PATH,
    user_rating=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    rating_scale=DEFAULT_RATING_SCALE,
    missing_rating_weight=DEFAULT_MISSING_RATING_WEIGHT,
    idf_weight_cap=DEFAULT_IDF_WEIGHT_CAP,
    confidence_smoothing=DEFAULT_CONFIDENCE_SMOOTHING,
):
    (
        tag_names,
        _,
        _,
        _,
        _,
        tag_score_vector,
    ) = build_user_tag_tfidf_components(
        handle,
        users_dir=users_dir,
        tag_stats_path=tag_stats_path,
        user_rating=user_rating,
        half_life_days=half_life_days,
        rating_scale=rating_scale,
        missing_rating_weight=missing_rating_weight,
        idf_weight_cap=idf_weight_cap,
        confidence_smoothing=confidence_smoothing,
    )

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


def select_diverse_problems(
    ranked_problems,
    limit=DEFAULT_RECOMMENDATION_LIMIT,
    primary_tag_quota=DEFAULT_PRIMARY_TAG_QUOTA,
):
    if limit <= 0:
        raise ValueError("limit must be positive")

    if primary_tag_quota <= 0:
        raise ValueError("primary_tag_quota must be positive")

    selected_problems = []
    primary_tag_counts = {}

    for problem in ranked_problems:
        primary_tag = problem.get("primary_tag")
        selected_count = primary_tag_counts.get(primary_tag, 0)
        if selected_count >= primary_tag_quota:
            continue

        selected_problem = dict(problem)
        selected_problem["recommendation_rank"] = len(selected_problems) + 1
        selected_problems.append(selected_problem)
        primary_tag_counts[primary_tag] = selected_count + 1

        if len(selected_problems) >= limit:
            break

    return selected_problems


def load_unsolved_problem_candidates(
    handle,
    users_dir=USER_DATA_DIR,
    problemset_path=CODEFORCES_PROBLEMS_PATH,
    tag_stats_path=TAG_STATS_PATH,
    min_rating_offset=DEFAULT_MIN_RATING_OFFSET,
    max_rating_offset=DEFAULT_MAX_RATING_OFFSET,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    rating_scale=DEFAULT_RATING_SCALE,
    missing_rating_weight=DEFAULT_MISSING_RATING_WEIGHT,
    idf_weight_cap=DEFAULT_IDF_WEIGHT_CAP,
    confidence_smoothing=DEFAULT_CONFIDENCE_SMOOTHING,
    recommendation_limit=DEFAULT_RECOMMENDATION_LIMIT,
    primary_tag_quota=DEFAULT_PRIMARY_TAG_QUOTA,
):
    submissions = load_user_submission_history(handle, users_dir)
    user_info = load_user_info(handle, users_dir)
    user_rating = user_info.get("rating")
    accepted_problem_keys = collect_accepted_problem_keys(submissions)
    problems = load_problemset(problemset_path)
    unsolved_problems = filter_accepted_problems(problems, accepted_problem_keys)
    candidate_problems = filter_problems_by_rating_range(
        unsolved_problems,
        user_rating,
        min_rating_offset=min_rating_offset,
        max_rating_offset=max_rating_offset,
    )
    tag_tfidf_score_map = load_tag_tfidf_score_map(
        handle,
        users_dir=users_dir,
        tag_stats_path=tag_stats_path,
        user_rating=user_rating,
        half_life_days=half_life_days,
        rating_scale=rating_scale,
        missing_rating_weight=missing_rating_weight,
        idf_weight_cap=idf_weight_cap,
        confidence_smoothing=confidence_smoothing,
    )
    ranked_candidate_problems = rank_problems_by_tfidf(
        candidate_problems,
        tag_tfidf_score_map,
    )
    recommended_problems = select_diverse_problems(
        ranked_candidate_problems,
        limit=recommendation_limit,
        primary_tag_quota=primary_tag_quota,
    )

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
