from cf_client.save_all_problem import save_all_problems
from data_pipeline.tag_extractor import extract_tag_stats


def run_preprocess_pipeline():
    raw_output_path, problem_count, statistic_count = save_all_problems()
    tag_stats_output_path, tag_stats = extract_tag_stats()

    return {
        "raw_problems_path": raw_output_path,
        "tag_stats_path": tag_stats_output_path,
        "problem_count": problem_count,
        "statistic_count": statistic_count,
        "tag_count": tag_stats["tag_count"],
    }


def main():
    result = run_preprocess_pipeline()

    print(
        "Saved "
        f"{result['problem_count']} problems and "
        f"{result['statistic_count']} statistics to "
        f"{result['raw_problems_path']}"
    )
    print(
        "Saved "
        f"{result['tag_count']} tag stats to "
        f"{result['tag_stats_path']}"
    )


if __name__ == "__main__":
    main()
