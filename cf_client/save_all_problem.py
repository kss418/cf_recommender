if __package__ is None or __package__ == "":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from cf_client.api_caller import ApiCaller
else:
    from .api_caller import ApiCaller

import json
from pathlib import Path


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "codeforces_problems.json"
)


def save_all_problems(output_path=DEFAULT_OUTPUT_PATH):
    caller = ApiCaller()
    result = caller.call_api("problemset.problems")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    problems = result.get("problems", [])
    statistics = result.get("problemStatistics", [])
    return output_path, len(problems), len(statistics)


def main():
    output_path, problem_count, statistic_count = save_all_problems()

    print(f"Saved {problem_count} problems and {statistic_count} statistics")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()
