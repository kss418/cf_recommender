from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cf_client.save_user_data import save_user_data
from recommender.problem_recommender import load_unsolved_problem_candidates

app = FastAPI(title="Codeforces Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_problem_url(problem):
    contest_id = problem.get("contestId")
    problem_index = problem.get("index")
    return f"https://codeforces.com/problemset/problem/{contest_id}/{problem_index}"


def format_problem_recommendation(problem, rank=None):
    return {
        "rank": rank if rank is not None else problem.get("recommendation_rank"),
        "contest_id": problem.get("contestId"),
        "index": problem.get("index"),
        "name": problem.get("name"),
        "rating": problem.get("rating"),
        "tags": problem.get("tags", []),
        "primary_tag": problem.get("primary_tag"),
        "tag_tfidf_score": problem.get("tfidf_score"),
        "effective_rating": problem.get("effective_rating"),
        "solve_probability": problem.get("solve_probability"),
        "difficulty_fit": problem.get("difficulty_fit"),
        "score": problem.get("final_score"),
        "url": build_problem_url(problem),
    }


def format_tag_skill(tag, stats, tag_tfidf_score_map):
    return {
        "tag": tag,
        "skill_rating": stats["skill_rating"],
        "skill_delta": stats["skill_delta"],
        "exposure": stats["exposure"],
        "solved_count": stats["solved_count"],
        "residual_average": stats["residual_average"],
        "confidence": stats["confidence"],
        "tag_tfidf_score": tag_tfidf_score_map.get(tag, 0.0),
    }


def build_tag_skills(recommendation_result):
    tag_tfidf_score_map = recommendation_result["tag_tfidf_score_map"]
    tag_skill_stats_map = recommendation_result["tag_skill_stats_map"]
    tag_skills = [
        format_tag_skill(tag, stats, tag_tfidf_score_map)
        for tag, stats in tag_skill_stats_map.items()
    ]
    return sorted(
        tag_skills,
        key=lambda tag_skill: tag_skill["skill_rating"],
    )


def build_problems_by_tag(recommendation_result):
    problems_by_tag = {}
    for problem in recommendation_result["ranked_candidate_problems"]:
        for tag in problem.get("tags", []):
            if tag not in recommendation_result["tag_skill_stats_map"]:
                continue

            problems_by_tag.setdefault(tag, [])
            if len(problems_by_tag[tag]) >= 5:
                continue

            problems_by_tag[tag].append(
                format_problem_recommendation(
                    problem,
                    rank=len(problems_by_tag[tag]) + 1,
                )
            )

    return problems_by_tag


@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    try:
        save_result = save_user_data(user_id)
        recommendation_result = load_unsolved_problem_candidates(user_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing local preprocessing data. "
                "Run preprocess_pipeline.py before requesting recommendations."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    return {
        "user_id": user_id,
        "user_data": {
            "output_path": str(save_result["output_path"]),
            "submission_count": save_result["submission_count"],
            "rating_change_count": save_result["rating_change_count"],
            "current_rating": save_result["current_rating"],
            "max_rating": save_result["max_rating"],
        },
        "user_rating": recommendation_result["user_rating"],
        "candidate_count": len(recommendation_result["candidate_problems"]),
        "tag_skills": build_tag_skills(recommendation_result),
        "problems_by_tag": build_problems_by_tag(recommendation_result),
        "recommendations": [
            format_problem_recommendation(problem)
            for problem in recommendation_result["recommended_problems"]
        ],
    }
