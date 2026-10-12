"""Small, simulator-independent CSV writer for WXAI controller telemetry."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACE_FIELDS = [
    "timestamp_utc",
    "phase",
    "phase_step",
    "command_index",
    "current_tcp_position_m",
    "target_tcp_position_m",
    "current_wrist_quaternion_wxyz",
    "target_quaternion_wxyz",
    "position_error_raw_m",
    "position_error_used_m",
    "orientation_error_raw_rad",
    "orientation_error_used_rad",
    "raw_max_joint_delta_rad",
    "applied_max_joint_delta_rad",
    "joint_delta_clamped",
    "joint_limit_clamped",
    "jacobian_min_singular_value",
    "jacobian_condition",
    "current_arm_joints",
    "target_arm_joints",
    "bread_position_m",
    "gripper_closed",
    "grasp_attached",
]


def _encode(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    return value


class ControlTraceWriter:
    """Write one flushed row per controller command so crashes keep evidence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=TRACE_FIELDS)
        self._writer.writeheader()
        self._handle.flush()

    def append(
        self,
        sample: dict[str, Any],
        *,
        phase: str,
        phase_step: int,
        **extra: Any,
    ) -> None:
        row: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "phase_step": phase_step,
            **sample,
            **extra,
        }
        self._writer.writerow({key: _encode(row.get(key)) for key in TRACE_FIELDS})
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "ControlTraceWriter":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
