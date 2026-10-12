"""Import processed serve-bread demonstrations into a portable session layout."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_sessions(inputs, output):
    output = Path(output).resolve()
    sessions = []
    for index, input_path in enumerate(inputs):
        source = Path(input_path).resolve()
        frames = source / "preprocess" / "all_data"
        if not frames.is_dir():
            raise FileNotFoundError(frames)
        name = f"session_{index:03d}"
        destination = output / name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite session: {destination}")
        records = []
        for annotation in sorted(frames.glob("[0-9]*/training_data.json")):
            data = json.loads(annotation.read_text())
            if "right" not in data.get("entities", {}).get("hands", {}):
                continue
            image = annotation.parent / "rgb_WoArm_WArmObjKpts.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            target = destination / "preprocess" / "all_data" / annotation.parent.name
            target.mkdir(parents=True)
            normalized = {
                "metadata": data["metadata"],
                "entities": {
                    "hands": data["entities"]["hands"],
                    "objects": data["entities"]["objects"],
                },
                "obs": {
                    "rgb_path": "observation.png",
                    "rgb_WoArm_WArmObjKpts_path": "observation.png",
                },
            }
            if "objects_kpts" in data.get("obs", {}):
                normalized["obs"]["objects_kpts"] = data["obs"]["objects_kpts"]
            normalized["metadata"]["dataset_kind"] = "imported_processed_reference"
            (target / "training_data.json").write_text(
                json.dumps(normalized, separators=(",", ":"), allow_nan=False) + "\n"
            )
            shutil.copy2(image, target / "observation.png")
            records.append({
                "frame": annotation.parent.name,
                "annotation_sha256_before_portable_paths": sha256(annotation),
                "observation_sha256": sha256(image),
            })
        if not records:
            raise ValueError(f"No valid training frames in {source}")
        entry = {
            "session": name,
            "split": "evaluation" if index == 0 else "training",
            "frame_count": len(records),
            "first_frame": records[0]["frame"],
            "last_frame": records[-1]["frame"],
            "dataset_kind": "imported_processed_reference",
            "geometry_unchanged": True,
            "frames": records,
        }
        (destination / "manifest.json").write_text(json.dumps(entry, indent=2) + "\n")
        sessions.append({key: value for key, value in entry.items() if key != "frames"})
        print(f"{name}: {len(records)} frames ({entry['split']})", flush=True)
    (output.parent / "manifest.json").write_text(
        json.dumps({"task": "serve_bread", "sessions": sessions}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-session", action="append", required=True)
    parser.add_argument("--output", default="data/serve_bread/demos")
    args = parser.parse_args()
    import_sessions(args.input_session, args.output)
