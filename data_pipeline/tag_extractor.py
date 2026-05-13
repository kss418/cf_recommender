if __package__ is None or __package__ == "":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

from collections import Counter

from data_pipeline.data_loader import get_data_path, load_json, write_json

INPUT_PATH = get_data_path("codeforces_problems")
TAG_STATS_OUTPUT_PATH = get_data_path("tag_stats")


def load_problems(input_path=INPUT_PATH):
    data = load_json(input_path)
    return data["problems"]


def count_problems_by_tag(problems):
    tag_counts = Counter()
    for problem in problems:
        tag_counts.update(set(problem.get("tags", [])))
    return tag_counts


def build_tag_stats(problems):
    tag_counts = count_problems_by_tag(problems)
    tags = [
        {
            "tag": tag,
            "problem_count": problem_count,
        }
        for tag, problem_count in sorted(tag_counts.items())
    ]

    return {
        "total_problem_count": len(problems),
        "tag_count": len(tag_counts),
        "problems_with_tags_count": sum(
            1 for problem in problems if problem.get("tags")
        ),
        "problems_without_tags_count": sum(
            1 for problem in problems if not problem.get("tags")
        ),
        "tags": tags,
    }


def extract_tag_stats(input_path=INPUT_PATH, output_path=TAG_STATS_OUTPUT_PATH):
    problems = load_problems(input_path)
    tag_stats = build_tag_stats(problems)
    output_path = write_json(output_path, tag_stats)
    return output_path, tag_stats


def main():
    output_path, tag_stats = extract_tag_stats()
    print(
        "Saved tag stats for "
        f"{tag_stats['total_problem_count']} problems to {output_path}"
    )


if __name__ == "__main__":
    main()
