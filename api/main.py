from fastapi import FastAPI, HTTPException

from cf_client.save_user_data import save_user_data
from recommender.problem_recommender import load_unsolved_problem_candidates

app = FastAPI(title="Codeforces Recommender API")


def build_problem_url(problem):
    contest_id = problem.get("contestId")
    problem_index = problem.get("index")
    return f"https://codeforces.com/problemset/problem/{contest_id}/{problem_index}"


def format_problem_recommendation(problem):
    return {
        "rank": problem["recommendation_rank"],
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
        "recommendations": [
            format_problem_recommendation(problem)
            for problem in recommendation_result["recommended_problems"]
        ],
    }
