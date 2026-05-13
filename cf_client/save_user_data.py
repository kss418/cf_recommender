if __package__ is None or __package__ == "":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from cf_client.api_caller import ApiCaller
else:
    from .api_caller import ApiCaller

import argparse

from data_pipeline.data_loader import get_data_path, write_json


DEFAULT_OUTPUT_DIR = get_data_path("users_dir")


def build_user_status_params(handle):
    return {"handle": handle}


def get_user_info(handle, caller):
    users = caller.call_api("user.info", {"handles": handle})
    if not users:
        raise RuntimeError(f"Codeforces user.info returned no user: {handle}")

    return users[0]


def get_user_rating_history(handle, caller):
    return caller.call_api("user.rating", {"handle": handle})


def get_user_submissions(handle, caller):
    params = build_user_status_params(handle)
    return caller.call_api("user.status", params)


def get_user_data(handle):
    caller = ApiCaller()
    user_info = get_user_info(handle, caller)
    rating_history = get_user_rating_history(handle, caller)
    submissions = get_user_submissions(handle, caller)

    return {
        "handle": handle,
        "info": user_info,
        "rating_history": rating_history,
        "submissions": submissions,
    }


def save_user_data(handle):
    user_data = get_user_data(handle)
    output_path = DEFAULT_OUTPUT_DIR / f"{handle}.json"
    output_path = write_json(output_path, user_data)

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
    args = parser.parse_args()

    result = save_user_data(args.handle)

    print(f"Saved {result['submission_count']} submissions")
    print(f"Saved {result['rating_change_count']} rating changes")
    print(f"Current rating: {result['current_rating']}")
    print(f"Max rating: {result['max_rating']}")
    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()
