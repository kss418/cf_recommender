import numpy as np
from fastapi import FastAPI, HTTPException, Query

from cf_client.save_user_data import save_user_data
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

app = FastAPI(title="Codeforces Recommender API")


def build_recommendations(
    user_id,
    top,
    half_life_days,
    rating_scale,
    idf_weight_cap,
    confidence_smoothing,
    missing_rating_weight,
):
    (
        tag_names,
        tag_tf_vector,
        tag_idf_vector,
        idf_weight_vector,
        confidence_vector,
        tag_score_vector,
    ) = build_user_tag_tfidf_components(
        user_id,
        half_life_days=half_life_days,
        rating_scale=rating_scale,
        idf_weight_cap=idf_weight_cap,
        confidence_smoothing=confidence_smoothing,
        missing_rating_weight=missing_rating_weight,
    )

    top_indices = np.argsort(tag_score_vector)[::-1][:top]
    recommendations = []
    for rank, index in enumerate(top_indices, start=1):
        recommendations.append(
            {
                "rank": rank,
                "tag": tag_names[index],
                "tf": float(tag_tf_vector[index]),
                "idf": float(tag_idf_vector[index]),
                "idf_weight": float(idf_weight_vector[index]),
                "confidence": float(confidence_vector[index]),
                "score": float(tag_score_vector[index]),
            }
        )

    return recommendations


@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    top: int = Query(default=10, ge=1, le=100),
    half_life_days: float = Query(default=DEFAULT_HALF_LIFE_DAYS, gt=0),
    rating_scale: float = Query(default=DEFAULT_RATING_SCALE, gt=0),
    idf_weight_cap: float = Query(default=DEFAULT_IDF_WEIGHT_CAP, gt=0),
    confidence_smoothing: float = Query(
        default=DEFAULT_CONFIDENCE_SMOOTHING,
        ge=0,
    ),
    missing_rating_weight: float = Query(
        default=DEFAULT_MISSING_RATING_WEIGHT,
        ge=0,
        le=1,
    ),
):
    try:
        save_result = save_user_data(user_id)
        recommendations = build_recommendations(
            user_id=user_id,
            top=top,
            half_life_days=half_life_days,
            rating_scale=rating_scale,
            idf_weight_cap=idf_weight_cap,
            confidence_smoothing=confidence_smoothing,
            missing_rating_weight=missing_rating_weight,
        )
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
        "top": top,
        "user_data": {
            "output_path": str(save_result["output_path"]),
            "submission_count": save_result["submission_count"],
            "rating_change_count": save_result["rating_change_count"],
            "current_rating": save_result["current_rating"],
            "max_rating": save_result["max_rating"],
        },
        "parameters": {
            "half_life_days": half_life_days,
            "rating_scale": rating_scale,
            "idf_weight_cap": idf_weight_cap,
            "confidence_smoothing": confidence_smoothing,
            "missing_rating_weight": missing_rating_weight,
        },
        "recommendations": recommendations,
    }
