"""Run the real learned checkpoint on one packaged demonstration frame."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/serve_bread/pretrained/latest.pt")
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/offline_prediction.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from inference.observation import first_recorded_frame, load_recorded_observation
    from inference.policy import ICTPolicy
    from inference.paths import display_path

    torch.set_num_threads(args.cpu_threads)
    checkpoint = args.checkpoint.resolve()
    cfg = {"ckpt": str(checkpoint), "seed": args.seed}
    if args.num_inference_steps is not None:
        cfg["num_inference_steps"] = args.num_inference_steps
    policy = ICTPolicy(cfg, device=args.device)
    frame = args.frame or first_recorded_frame(ROOT / "data/serve_bread/demos/session_000/preprocess/all_data")
    observation = load_recorded_observation(frame, policy.training_config.get("img_name", "observation.png"))
    anchor = observation.objects[observation.anchor_key]
    tokens, mask = policy.build_ict(observation.hands_in_cam, observation.grippers, observation.objects, observation.anchor_key)
    rgb = policy.prepare_image(observation.rgb_bgr)
    uv = policy.compute_anchor_uv(observation.image_anchor, observation.K, observation.rgb_bgr.shape[1], observation.rgb_bgr.shape[0])
    trajectory, done = policy.infer(rgb, tokens, mask, uv)
    arrays = {"ict": tokens.cpu().numpy(), "ict_mask": mask.cpu().numpy(),
              "input_rgb_bgr": observation.rgb_bgr, "K": observation.K,
              "initial_noise": policy.noise.cpu().numpy(), "done_probability": np.array(done)}
    for side, (position, rotation, grasp) in trajectory.items():
        hand_poses = np.asarray([policy.decode_ee_in_cam(p, r, anchor, np.eye(4, dtype=np.float32))
                                 for p, r in zip(position, rotation)])
        arrays.update({f"{side}_position": position, f"{side}_o6d": rotation, f"{side}_grasp": grasp,
                       f"{side}_hand_pose_in_cam": hand_poses})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {"project": "Human2Robot", "inference_mode": "learned_flow_matching",
                "checkpoint": display_path(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "frame": display_path(observation.source), "device": args.device,
                "num_inference_steps": policy.num_inference_steps, "seed": args.seed,
                "trajectory_frame": policy.frame_mode, "done_probability": done,
                "position_normalization": {"mean": policy.pos_mean.tolist(), "std": policy.pos_std.tolist()},
                "anchor_key": observation.anchor_key, "checkpoint_loaded_strict": True,
                "observation_mode": "recorded_demonstration", "physical_execution": False,
                "trajectory": display_path(output), **policy.checkpoint_metadata}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved learned trajectory: {output}")
    return 0


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
