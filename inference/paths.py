"""Repository-relative artifact labels without depending on the launch cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display_path(value):
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def portable_text(value):
    return str(value).replace(str(ROOT) + "/", "").replace(str(ROOT), ".")


def portable_record(value):
    if isinstance(value, str):
        return portable_text(value)
    if isinstance(value, dict):
        return {key: portable_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_record(item) for item in value]
    return value
