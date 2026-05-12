import json
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("data/raw/codeforces_problems.json")
TAG_STATS_OUTPUT_PATH = Path("data/processed/codeforces_tag_stats.json")


def load_problems(input_path=INPUT_PATH):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
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


def write_json(output_path, value):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    problems = load_problems()
    tag_stats = build_tag_stats(problems)

    write_json(TAG_STATS_OUTPUT_PATH, tag_stats)

    print(
        "Saved tag stats for "
        f"{tag_stats['total_problem_count']} problems to {TAG_STATS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
