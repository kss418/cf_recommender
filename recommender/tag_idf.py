import json
from pathlib import Path

import numpy as np

TAG_STATS_PATH = Path("data/processed/codeforces_tag_stats.json")


def load_tag_stats(tag_stats_path=TAG_STATS_PATH):
    return json.loads(Path(tag_stats_path).read_text(encoding="utf-8"))


def extract_tag_counts(tag_stats):
    total_problem_count = tag_stats["total_problem_count"]
    tag_problem_counts = {
        row["tag"]: row["problem_count"]
        for row in tag_stats["tags"]
    }
    return total_problem_count, tag_problem_counts


def load_tag_counts(tag_stats_path=TAG_STATS_PATH):
    tag_stats = load_tag_stats(tag_stats_path)
    return extract_tag_counts(tag_stats)


def build_tag_count_vector(tag_problem_counts):
    tag_names = sorted(tag_problem_counts)
    tag_problem_count_vector = np.array(
        [tag_problem_counts[tag] for tag in tag_names],
        dtype=np.int64,
    )
    return tag_names, tag_problem_count_vector


def load_tag_count_vector(tag_stats_path=TAG_STATS_PATH):
    total_problem_count, tag_problem_counts = load_tag_counts(tag_stats_path)
    tag_names, tag_problem_count_vector = build_tag_count_vector(tag_problem_counts)
    return total_problem_count, tag_names, tag_problem_count_vector


def calculate_idf_vector(total_problem_count, tag_problem_count_vector):
    tag_problem_count_vector = np.asarray(tag_problem_count_vector, dtype=np.float64)
    return np.log(
        (total_problem_count + 1) / (tag_problem_count_vector + 1)
    ) + 1


def load_tag_idf_vector(tag_stats_path=TAG_STATS_PATH):
    total_problem_count, tag_names, tag_problem_count_vector = load_tag_count_vector(
        tag_stats_path
    )
    tag_idf_vector = calculate_idf_vector(
        total_problem_count,
        tag_problem_count_vector,
    )
    return total_problem_count, tag_names, tag_problem_count_vector, tag_idf_vector
