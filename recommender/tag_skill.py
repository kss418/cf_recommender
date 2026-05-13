import numpy as np

from recommender.submission_history import (
    analyze_submissions_by_problem,
    load_user_info,
    load_user_submissions,
)
from recommender.tag_tf import DEFAULT_RATING_SCALE, calculate_weakness_signal

DEFAULT_SKILL_DELTA_LIMIT = 300
DEFAULT_SKILL_CONFIDENCE_SMOOTHING = 3.0
DEFAULT_SKILL_PRIOR_SCALE = 400
DEFAULT_SKILL_SEARCH_MARGIN = 1200
DEFAULT_SKILL_SEARCH_ITERATIONS = 80
DEFAULT_TARGET_SOLVE_PROBABILITY = 0.45
DEFAULT_SOLVE_PROBABILITY_SIGMA = 0.15


def calculate_expected_solve_probability(user_rating, problem_rating):
    if user_rating is None or problem_rating is None:
        raise ValueError("user_rating and problem_rating are required")

    rating_gap = (problem_rating - user_rating) / DEFAULT_RATING_SCALE
    return float(1 / (1 + np.power(10, rating_gap)))


def calculate_skill_residual(actual_result, expected_result):
    return actual_result - expected_result


def calculate_actual_skill_result(problem_analysis):
    return 1.0 - calculate_weakness_signal(problem_analysis)


def calculate_tag_skill_confidence(exposure):
    denominator = exposure + DEFAULT_SKILL_CONFIDENCE_SMOOTHING
    if denominator <= 0:
        return 0.0

    return exposure / denominator


def calculate_difficulty_fit(solve_probability):
    probability_gap = solve_probability - DEFAULT_TARGET_SOLVE_PROBABILITY
    denominator = 2 * DEFAULT_SOLVE_PROBABILITY_SIGMA ** 2
    return float(np.exp(-(probability_gap ** 2) / denominator))


def calculate_skill_rating_gradient(skill_rating, problem_results, user_rating):
    rating_prior_gradient = (
        -(skill_rating - user_rating) / (DEFAULT_SKILL_PRIOR_SCALE ** 2)
    )
    result_gradient = 0.0

    for problem_rating, actual_result in problem_results:
        expected_result = calculate_expected_solve_probability(
            skill_rating,
            problem_rating,
        )
        result_gradient += actual_result - expected_result

    result_gradient *= np.log(10) / DEFAULT_RATING_SCALE
    return result_gradient + rating_prior_gradient


def estimate_tag_skill_rating(problem_results, user_rating):
    problem_ratings = [
        problem_rating
        for problem_rating, _ in problem_results
    ]
    lower_rating = min(
        min(problem_ratings) - DEFAULT_SKILL_SEARCH_MARGIN,
        user_rating - DEFAULT_SKILL_SEARCH_MARGIN,
    )
    upper_rating = max(
        max(problem_ratings) + DEFAULT_SKILL_SEARCH_MARGIN,
        user_rating + DEFAULT_SKILL_SEARCH_MARGIN,
    )

    for _ in range(DEFAULT_SKILL_SEARCH_ITERATIONS):
        middle_rating = (lower_rating + upper_rating) / 2
        gradient = calculate_skill_rating_gradient(
            middle_rating,
            problem_results,
            user_rating,
        )

        if gradient > 0:
            lower_rating = middle_rating
        else:
            upper_rating = middle_rating

    return float((lower_rating + upper_rating) / 2)


def calculate_average_residual(problem_results, skill_rating):
    residuals = [
        calculate_skill_residual(
            actual_result,
            calculate_expected_solve_probability(skill_rating, problem_rating),
        )
        for problem_rating, actual_result in problem_results
    ]
    return sum(residuals) / len(residuals)


def build_tag_skill_stats_map(problem_analysis_by_key, user_rating):
    if user_rating is None:
        raise ValueError("User rating is required for tag skill estimation")

    tag_problem_results = {}

    for problem_analysis in problem_analysis_by_key.values():
        problem = problem_analysis.get("problem", {})
        problem_rating = problem.get("rating")
        problem_tags = set(problem.get("tags", []))

        if problem_rating is None or not problem_tags:
            continue

        actual_result = calculate_actual_skill_result(problem_analysis)

        for tag in problem_tags:
            tag_problem_results.setdefault(tag, []).append(
                (problem_rating, actual_result)
            )

    tag_skill_stats_map = {}
    for tag in sorted(tag_problem_results):
        problem_results = tag_problem_results[tag]
        exposure = len(problem_results)
        solved_count = sum(
            actual_result
            for _, actual_result in problem_results
        )
        skill_rating = estimate_tag_skill_rating(problem_results, user_rating)
        residual_average = calculate_average_residual(problem_results, user_rating)
        confidence = calculate_tag_skill_confidence(exposure)
        skill_delta = float(
            np.clip(
                skill_rating - user_rating,
                -DEFAULT_SKILL_DELTA_LIMIT,
                DEFAULT_SKILL_DELTA_LIMIT,
            )
        )

        tag_skill_stats_map[tag] = {
            "exposure": exposure,
            "solved_count": int(solved_count),
            "residual_average": float(residual_average),
            "confidence": float(confidence),
            "skill_rating": skill_rating,
            "skill_delta": skill_delta,
        }

    return tag_skill_stats_map


def build_user_tag_skill_stats_map(handle):
    user_rating = load_user_info(handle).get("rating")
    submissions = load_user_submissions(handle)
    problem_analysis_by_key = analyze_submissions_by_problem(submissions)
    return build_tag_skill_stats_map(problem_analysis_by_key, user_rating)
