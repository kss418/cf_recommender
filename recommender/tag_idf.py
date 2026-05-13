import numpy as np

from data_pipeline.data_loader import get_data_path, load_json

TAG_STATS_PATH = get_data_path("tag_stats")


def load_tag_stats():
    return load_json(TAG_STATS_PATH)


def extract_tag_counts(tag_stats):
    total_problem_count = tag_stats["total_problem_count"]
    tag_problem_counts = {
        row["tag"]: row["problem_count"]
        for row in tag_stats["tags"]
    }
    return total_problem_count, tag_problem_counts


def load_tag_counts():
    tag_stats = load_tag_stats()
    return extract_tag_counts(tag_stats)


def build_tag_count_vector(tag_problem_counts):
    tag_names = sorted(tag_problem_counts)
    tag_problem_count_vector = np.array(
        [tag_problem_counts[tag] for tag in tag_names],
        dtype=np.int64,
    )
    return tag_names, tag_problem_count_vector


def load_tag_count_vector():
    total_problem_count, tag_problem_counts = load_tag_counts()
    tag_names, tag_problem_count_vector = build_tag_count_vector(tag_problem_counts)
    return total_problem_count, tag_names, tag_problem_count_vector


def calculate_idf_vector(total_problem_count, tag_problem_count_vector):
    tag_problem_count_vector = np.asarray(tag_problem_count_vector, dtype=np.float64)
    return np.log(
        (total_problem_count + 1) / (tag_problem_count_vector + 1)
    ) + 1


def load_tag_idf_vector():
    total_problem_count, tag_names, tag_problem_count_vector = load_tag_count_vector()
    tag_idf_vector = calculate_idf_vector(
        total_problem_count,
        tag_problem_count_vector,
    )
    return total_problem_count, tag_names, tag_problem_count_vector, tag_idf_vector
