"""Physical, contact-distance-gated bread grasp support for the WXAI demo.

The Human2Robot object latch only updates policy state. This module supplies the
corresponding Isaac physics operation: a measured-relative-pose FixedJoint is
created when the gripper closes near the bread, and removed when it opens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from isaacsim.core.experimental.prims import RigidPrim
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        [
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ]
    ).as_matrix()
    transform[:3, 3] = position
    return transform


def _gf_quaternion(rotation: np.ndarray) -> Gf.Quatf:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*map(float, xyzw[:3])))


def ensure_wxai_tcp_site(
    stage,
    robot_root: str,
    offset: tuple[float, float, float] = (0.1055, 0.0, 0.0),
) -> str:
    """Author the Trossen controller's logical TCP as an inspectable SiteAPI."""
    link_path = f"{robot_root}/link_6"
    if not stage.GetPrimAtPath(link_path).IsValid():
        raise RuntimeError(f"WXAI end-effector link is missing: {link_path}")
    site_path = f"{link_path}/wxai_tcp"
    site = UsdGeom.Xform.Define(stage, site_path)
    site.GetPrim().AddAppliedSchema("IsaacSiteAPI")
    xform = UsdGeom.Xformable(site.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, offset)))
    return site_path


def prepare_bread_physics(stage, bread_root: str) -> str:
    """Return the bread rigid-body path, authoring a body only when absent."""
    root = stage.GetPrimAtPath(bread_root)
    if not root.IsValid():
        raise RuntimeError(f"Bread prim is missing: {bread_root}")

    rigid_prims = [
        prim
        for prim in stage.Traverse()
        if _under(str(prim.GetPath()), bread_root)
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if root.HasAPI(UsdPhysics.RigidBodyAPI):
        body = root
    elif rigid_prims:
        body = rigid_prims[0]
    else:
        body_api = UsdPhysics.RigidBodyAPI.Apply(root)
        body_api.CreateRigidBodyEnabledAttr().Set(True)
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr().Set(0.08)
        body = root

    collision_prims = [
        prim
        for prim in stage.Traverse()
        if _under(str(prim.GetPath()), bread_root)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not collision_prims:
        raise RuntimeError(
            f"Bread has no collision geometry beneath {bread_root}; cannot grasp it"
        )
    return str(body.GetPath())


def apply_grasp_friction(
    stage,
    bread_root: str,
    robot_root: str,
    material_path: str = "/World/PhysicsMaterials/wxai_grasp_high_friction",
) -> int:
    """Bind a high-friction material to bread and finger collision prims."""
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(1.0)
    physics_material.CreateDynamicFrictionAttr().Set(0.9)
    physics_material.CreateRestitutionAttr().Set(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set("max")
    physx_material.CreateRestitutionCombineModeAttr().Set("min")

    finger_roots = (
        f"{robot_root}/gripper_left",
        f"{robot_root}/gripper_right",
    )
    bound = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        relevant = _under(path, bread_root) or any(
            _under(path, finger_root) for finger_root in finger_roots
        )
        if not relevant or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
        bound += 1
    return bound


class AssistedBreadGrasp:
    """Attach bread at measured relative pose and release under gravity."""

    def __init__(
        self,
        stage,
        bread_body_path: str,
        robot_link_path: str,
        tcp_pose_getter,
        output_dir: Path,
        attach_distance: float = 0.14,
        enabled: bool = True,
    ):
        self.stage = stage
        self.bread_body_path = bread_body_path
        self.robot_link_path = robot_link_path
        self.tcp_pose_getter = tcp_pose_getter
        self.attach_distance = float(attach_distance)
        self.enabled = bool(enabled)
        self.joint_path = "/World/Runtime/wxai_bread_grasp"
        self.bread = RigidPrim(bread_body_path)
        self.robot_link = RigidPrim(robot_link_path)
        self.attached = False
        self.assist_used = False
        self.attach_count = 0
        self.release_count = 0
        self.steps_since_release = 0
        self.max_bread_z = float("-inf")
        self.initial_bread_z: float | None = None
        self.log_path = Path(output_dir) / "grasp_events.log"
        self.log_path.write_text("", encoding="utf-8")

        self.sample()
        self.initial_bread_z = self.max_bread_z

    @staticmethod
    def _world_pose(rigid_prim: RigidPrim) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = rigid_prim.get_world_poses()
        return (
            np.asarray(position.numpy(), dtype=np.float64)[0],
            np.asarray(orientation.numpy(), dtype=np.float64)[0],
        )

    def _log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(message)

    def sample(self) -> np.ndarray:
        position, _ = self._world_pose(self.bread)
        self.max_bread_z = max(self.max_bread_z, float(position[2]))
        return position

    def bread_world_pose(self) -> np.ndarray:
        position, orientation = self._world_pose(self.bread)
        self.max_bread_z = max(self.max_bread_z, float(position[2]))
        return _pose_matrix(position, orientation)

    def distance_to_tcp(self) -> float:
        bread_position = self.sample()
        tcp_position = np.asarray(self.tcp_pose_getter(), dtype=np.float64)[:3, 3]
        return float(np.linalg.norm(bread_position - tcp_position))

    def update(self, commanded_closed: bool) -> None:
        self.sample()
        if not self.enabled:
            return
        if not commanded_closed:
            if self.assist_used and not self.attached:
                self.steps_since_release += 1
            self.release()
            return
        if self.attached:
            self.steps_since_release = 0
            return

        distance = self.distance_to_tcp()
        if distance > self.attach_distance:
            return

        link_position, link_orientation = self._world_pose(self.robot_link)
        bread_position, bread_orientation = self._world_pose(self.bread)
        world_from_link = _pose_matrix(link_position, link_orientation)
        world_from_bread = _pose_matrix(bread_position, bread_orientation)

        # Put the joint frame at the bread body origin. The measured local0
        # transform preserves the current relative pose and prevents snapping.
        link_from_joint = np.linalg.inv(world_from_link) @ world_from_bread
        joint = UsdPhysics.FixedJoint.Define(self.stage, self.joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.robot_link_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self.bread_body_path)])
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*map(float, link_from_joint[:3, 3]))
        )
        joint.CreateLocalRot0Attr().Set(_gf_quaternion(link_from_joint[:3, :3]))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        self.attached = True
        self.assist_used = True
        self.attach_count += 1
        self.steps_since_release = 0
        self._log(
            f"ATTACH distance_m={distance:.6f} body={self.bread_body_path} "
            f"link={self.robot_link_path}"
        )

    def release(self) -> None:
        if not self.attached:
            return
        self.stage.RemovePrim(self.joint_path)
        self.attached = False
        self.release_count += 1
        self.steps_since_release = 0
        self._log("RELEASE")

    @property
    def lift_m(self) -> float:
        if self.initial_bread_z is None:
            return 0.0
        return self.max_bread_z - self.initial_bread_z
