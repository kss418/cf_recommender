from recommender.submission_history import USER_DATA_DIR
from recommender.tag_idf import TAG_STATS_PATH, load_tag_idf_vector
from recommender.tag_tf import DEFAULT_HALF_LIFE_DAYS, build_user_tag_tf_vector


def build_user_tag_tfidf_vector(
    handle,
    users_dir=USER_DATA_DIR,
    tag_stats_path=TAG_STATS_PATH,
    current_time_seconds=None,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    normalize_tf=True,
):
    _, tag_names, _, tag_idf_vector = load_tag_idf_vector(tag_stats_path)
    tag_tf_vector = build_user_tag_tf_vector(
        handle,
        tag_names,
        users_dir=users_dir,
        current_time_seconds=current_time_seconds,
        half_life_days=half_life_days,
        normalize=normalize_tf,
    )
    tag_tfidf_vector = tag_tf_vector * tag_idf_vector

    return tag_names, tag_tf_vector, tag_idf_vector, tag_tfidf_vector
