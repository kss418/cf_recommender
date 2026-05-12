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
    / "users"
)


def build_user_status_params(handle, from_index=None, count=None):
    params = {"handle": handle}

    if from_index is not None:
        params["from"] = from_index

    if count is not None:
        params["count"] = count

    return params


def get_user_info(handle, caller=None):
    if caller is None:
        caller = ApiCaller()

    users = caller.call_api("user.info", {"handles": handle})
    if not users:
        raise RuntimeError(f"Codeforces user.info returned no user: {handle}")

    return users[0]


def get_user_rating_history(handle, caller=None):
    if caller is None:
        caller = ApiCaller()

    return caller.call_api("user.rating", {"handle": handle})


def get_user_submissions(handle, from_index=None, count=None, caller=None):
    if caller is None:
        caller = ApiCaller()

    params = build_user_status_params(handle, from_index, count)
    return caller.call_api("user.status", params)


def get_user_data(handle, from_index=None, count=None):
    caller = ApiCaller()
    user_info = get_user_info(handle, caller)
    rating_history = get_user_rating_history(handle, caller)
    submissions = get_user_submissions(
        handle,
        from_index=from_index,
        count=count,
        caller=caller,
    )

    return {
        "handle": handle,
        "info": user_info,
        "rating_history": rating_history,
        "submissions": submissions,
    }


def save_user_data(
    handle,
    output_path=None,
    from_index=None,
    count=None,
):
    user_data = get_user_data(
        handle,
        from_index=from_index,
        count=count,
    )

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{handle}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(user_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_path": output_path,
        "submission_count": len(user_data["submissions"]),
        "rating_change_count": len(user_data["rating_history"]),
        "current_rating": user_data["info"].get("rating"),
        "max_rating": user_data["info"].get("maxRating"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("handle")
    parser.add_argument("--from", dest="from_index", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = save_user_data(
        args.handle,
        output_path=args.output,
        from_index=args.from_index,
        count=args.count,
    )

    print(f"Saved {result['submission_count']} submissions")
    print(f"Saved {result['rating_change_count']} rating changes")
    print(f"Current rating: {result['current_rating']}")
    print(f"Max rating: {result['max_rating']}")
    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()
