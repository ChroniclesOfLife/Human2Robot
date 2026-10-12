"""Audit portable paths, serve-bread data, weights, and learned-only entrypoints."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check(root, hashes=False):
    errors = []
    for name in ("training", "utils", "preprocess", "inference", "scripts", "cfg", "assets", "scenes", "calibration", "data", "runs"):
        if not (root / name).is_dir():
            errors.append(f"Missing directory: {name}")
    text_types = {".py", ".sh", ".yaml", ".yml", ".json", ".usda"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in {"__pycache__", ".git", ".venv", ".pytest_cache", "outputs"} for part in relative.parts):
            continue
        if relative.parts[0] == "licenses" or path.name == "THIRD_PARTY_NOTICES.md":
            continue
        if path.suffix in text_types:
            text = path.read_text(encoding="utf-8")
            if re.search(r"/(?:home|Users)/[^/]+/", text):
                errors.append(f"Machine-specific path: {relative}")
    task_dirs = sorted(path.name for path in (root / "data").iterdir() if path.is_dir())
    if task_dirs != ["serve_bread"]:
        errors.append(f"Unexpected tasks: {task_dirs}")
    total = 0
    manifest = json.loads((root / "data/serve_bread/manifest.json").read_text())
    for session in manifest["sessions"]:
        directory = root / "data/serve_bread/demos" / session["session"]
        entry = json.loads((directory / "manifest.json").read_text())
        annotations = sorted((directory / "preprocess/all_data").glob("*/training_data.json"))
        if len(annotations) != entry["frame_count"]:
            errors.append(f"Frame-count mismatch: {session['session']}")
        records = {item["frame"]: item for item in entry["frames"]}
        for annotation in annotations:
            data = json.loads(annotation.read_text())
            image = annotation.parent / "observation.png"
            if not image.is_file():
                errors.append(f"Missing observation: {annotation.relative_to(root)}")
            if "right" not in data["entities"]["hands"] or not {"obj1", "obj2"}.issubset(data["entities"]["objects"]):
                errors.append(f"Missing hand/object labels: {annotation.relative_to(root)}")
            if hashes and image.is_file() and digest(image) != records[annotation.parent.name]["observation_sha256"]:
                errors.append(f"Image checksum mismatch: {image.relative_to(root)}")
            total += 1
    pretrained = root / "runs/serve_bread/pretrained"
    source = json.loads((pretrained / "provenance.json").read_text())
    if not source["all_model_tensors_identical_after_export"]:
        errors.append("Model tensors changed")
    if hashes and digest(pretrained / "latest.pt") != source["exported_checkpoint_sha256"]:
        errors.append("Checkpoint checksum mismatch")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {total} labeled serve-bread frames, portable runtime paths, pretrained model.")
    print("PASS: only the serve-bread task is included.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--hashes", action="store_true", help="Verify all copied image and model checksums.")
    arguments = parser.parse_args()
    check(arguments.root.resolve(), arguments.hashes)
