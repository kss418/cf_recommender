if __package__ is None or __package__ == "":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from cf_client.api_caller import ApiCaller
else:
    from .api_caller import ApiCaller

from data_pipeline.data_loader import get_data_path, write_json


DEFAULT_OUTPUT_PATH = get_data_path("codeforces_problems")


def save_all_problems(output_path=DEFAULT_OUTPUT_PATH):
    caller = ApiCaller()
    result = caller.call_api("problemset.problems")

    output_path = write_json(output_path, result)

    problems = result.get("problems", [])
    statistics = result.get("problemStatistics", [])
    return output_path, len(problems), len(statistics)


def main():
    output_path, problem_count, statistic_count = save_all_problems()

    print(f"Saved {problem_count} problems and {statistic_count} statistics")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
