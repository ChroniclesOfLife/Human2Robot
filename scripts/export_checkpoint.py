"""Export exact model weights with portable metadata and no optimizer baggage."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(source, output, overwrite=False):
    source, output = Path(source), Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    original = torch.load(source, map_location="cpu", weights_only=True)
    cfg = original["cfg"].copy()
    cfg = {
        key: value for key, value in cfg.items()
        if key not in {"out_dir", "data_root", "data_sources", "eval_source", "hand_tracking_method"}
        and not key.endswith("PATHS_TRAIN") and not key.endswith("PATHS_EVAL")
    }
    cfg.update({
        "out_dir": "runs/serve_bread/pretrained",
        "data_root": "data",
        "SESSIONS_TRAIN": ["data/serve_bread/demos/session_001"],
        "SESSIONS_EVAL": ["data/serve_bread/demos/session_000"],
        "data_sources": {"demos": 1},
        "eval_source": "demos",
        "hand_tracking_method": "keypoints",
        "img_name": "observation.png",
        "checkpoint_kind": "imported_pretrained_model",
        "job": "Human2Robot",
    })
    exported = {key: original[key] for key in ("epoch", "model", "best", "global_step") if key in original}
    exported["cfg"] = cfg
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    restored = torch.load(output, map_location="cpu", weights_only=True)
    if original["model"].keys() != restored["model"].keys():
        raise AssertionError("Model keys changed during export")
    for key, value in original["model"].items():
        if not torch.equal(value, restored["model"][key]):
            raise AssertionError(f"Model tensor changed: {key}")
    (output.parent / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    provenance = {
        "kind": "imported_pretrained_model",
        "epoch": original.get("epoch"),
        "global_step": original.get("global_step"),
        "model_tensor_count": len(original["model"]),
        "all_model_tensors_identical_after_export": True,
        "optimizer_state_included": False,
        "source_checkpoint_sha256": sha256(source),
        "exported_checkpoint_sha256": sha256(output),
        "changes": "Portable metadata and image filename; model tensors unchanged.",
    }
    (output.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="runs/serve_bread/pretrained/latest.pt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_checkpoint(args.input, args.output, args.overwrite)
