import json
from collections import Counter
from pathlib import Path

INPUT_PATH = Path("data/raw/codeforces_problems.json")
TAGS_OUTPUT_PATH = Path("data/processed/codeforces_tags.json")
TAG_STATS_OUTPUT_PATH = Path("data/processed/codeforces_tag_stats.json")


def load_problems(input_path=INPUT_PATH):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return data["problems"]


def extract_tags(problems):
    return sorted({
        tag
        for problem in problems
        for tag in problem.get("tags", [])
    })


def count_problems_by_tag(problems):
    tag_counts = Counter()
    for problem in problems:
        tag_counts.update(set(problem.get("tags", [])))
    return dict(sorted(tag_counts.items()))


def build_tag_stats(problems):
    tag_counts = count_problems_by_tag(problems)
    return {
        "total_problem_count": len(problems),
        "tag_count": len(tag_counts),
        "problems_with_tags_count": sum(
            1 for problem in problems if problem.get("tags")
        ),
        "problems_without_tags_count": sum(
            1 for problem in problems if not problem.get("tags")
        ),
        "tag_problem_counts": tag_counts,
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
    tags = extract_tags(problems)
    tag_stats = build_tag_stats(problems)

    write_json(TAGS_OUTPUT_PATH, tags)
    write_json(TAG_STATS_OUTPUT_PATH, tag_stats)

    print(f"Saved {len(tags)} tags to {TAGS_OUTPUT_PATH}")
    print(
        "Saved tag stats for "
        f"{tag_stats['total_problem_count']} problems to {TAG_STATS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
