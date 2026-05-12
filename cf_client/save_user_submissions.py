if __package__ is None or __package__ == "":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from cf_client.api_caller import ApiCaller
else:
    from .api_caller import ApiCaller

import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "user_submissions"
)


def build_user_status_params(handle, from_index=None, count=None):
    params = {"handle": handle}

    if from_index is not None:
        params["from"] = from_index

    if count is not None:
        params["count"] = count

    return params


def get_user_submissions(handle, from_index=None, count=None):
    caller = ApiCaller()
    params = build_user_status_params(handle, from_index, count)
    return caller.call_api("user.status", params)


def save_user_submissions(
    handle,
    output_path=None,
    from_index=None,
    count=None,
):
    submissions = get_user_submissions(
        handle,
        from_index=from_index,
        count=count,
    )

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{handle}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(submissions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path, len(submissions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("handle")
    parser.add_argument("--from", dest="from_index", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_path, submission_count = save_user_submissions(
        args.handle,
        output_path=args.output,
        from_index=args.from_index,
        count=args.count,
    )

    print(f"Saved {submission_count} submissions")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
