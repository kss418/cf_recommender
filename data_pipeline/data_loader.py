import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "data_paths.json"


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_data_config(config_path=DEFAULT_CONFIG_PATH):
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("Data config JSON must contain an object")
    return config


def get_data_path(path_key, config_path=DEFAULT_CONFIG_PATH):
    config = load_data_config(config_path)
    paths = config.get("paths", {})
    if path_key not in paths:
        raise KeyError(f"Data config missing path: {path_key}")
    return resolve_project_path(paths[path_key])


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_json_data(path_key, config_path=DEFAULT_CONFIG_PATH):
    return load_json(get_data_path(path_key, config_path))


def write_json_data(path_key, value, config_path=DEFAULT_CONFIG_PATH):
    return write_json(get_data_path(path_key, config_path), value)


def get_user_data_path(handle, config_path=DEFAULT_CONFIG_PATH):
    return get_data_path("users_dir", config_path) / f"{handle}.json"


def load_user_data(handle, config_path=DEFAULT_CONFIG_PATH):
    return load_json(get_user_data_path(handle, config_path))


def write_user_data(handle, user_data, config_path=DEFAULT_CONFIG_PATH):
    return write_json(get_user_data_path(handle, config_path), user_data)
