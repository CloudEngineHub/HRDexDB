"""Self-contained HRDexDB geometry, pose, and robot-state loaders."""

from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_DATASET_ROOT = ""  # put the dataset path

ROBOT_URDFS = {
    "allegro": "xarm_allegro.urdf",
    "allegro_v5": "allegro_v5/xarm_allegro_v5.urdf",
    "inspire": "xarm_inspire_DFTP.urdf",
    "inspire_f1": "xarm_inspire_f1_right.urdf",
    "inspire_new": "xarm_inspire_DFTP.urdf",
}


class ExportError(RuntimeError):
    """Raised when an episode cannot be processed."""


@dataclass
class MeshData:
    name: str
    vertices: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None = None
    mode: int = 4


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    qpos_index: int | None = None
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class VisualSpec:
    link: str
    mesh_path: Path
    origin: np.ndarray
    scale: np.ndarray
    material: str | None


@dataclass
class UrdfModel:
    root_link: str
    joints: list[JointSpec]
    visuals: list[VisualSpec]
    materials: dict[str, tuple[float, float, float, float]]
    child_joints: dict[str, list[JointSpec]] = field(default_factory=dict)


_MESH_CACHE: dict[Path, MeshData] = {}
_URDF_CACHE: dict[Path, UrdfModel] = {}


def parse_vec(value: str | None, default: Sequence[float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in value.split()], dtype=float)


def urdf_rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def make_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = urdf_rpy_matrix(rpy)
    out[:3, 3] = np.asarray(xyz, dtype=float)
    return out


def apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def axis_angle_transform(axis: np.ndarray, value: float, joint_type: str) -> np.ndarray:
    out = np.eye(4, dtype=float)
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return out
    axis = axis / norm
    if joint_type in {"revolute", "continuous"}:
        out[:3, :3] = axis_angle_matrix(axis, float(value))
    elif joint_type == "prismatic":
        out[:3, 3] = axis * float(value)
    return out


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 0 else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=float)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def orthonormalized(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def load_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=True), dtype=float)


def load_series(data_dir: Path, candidates: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    for name in candidates:
        data_path = data_dir / name
        if not data_path.exists():
            continue
        data = load_array(data_path)
        time_candidates = [
            data_path.with_name(data_path.stem + "_time.npy"),
            data_dir / "time.npy",
        ]
        for time_path in time_candidates:
            if time_path.exists():
                times = load_array(time_path).reshape(-1)
                n = min(len(times), len(data))
                return data[:n], times[:n]
        return data, np.arange(len(data), dtype=float)
    raise ExportError(f"none of {tuple(candidates)} found in {data_dir}")


def resample_to(times_src: np.ndarray, data_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    times_src = np.asarray(times_src, dtype=float).reshape(-1)
    data_src = np.asarray(data_src, dtype=float)
    times_dst = np.asarray(times_dst, dtype=float).reshape(-1)
    n = min(len(times_src), len(data_src))
    times_src = times_src[:n]
    data_src = data_src[:n]
    if n == 0:
        raise ExportError("cannot resample an empty time series")
    if n == 1:
        return np.repeat(data_src[:1], len(times_dst), axis=0)
    order = np.argsort(times_src)
    times_src = times_src[order]
    data_src = data_src[order]
    flat = data_src.reshape(len(data_src), -1)
    out = np.stack(
        [np.interp(times_dst, times_src, flat[:, i]) for i in range(flat.shape[1])],
        axis=1,
    )
    return out.reshape((len(times_dst),) + data_src.shape[1:])


def resample_poses(poses: np.ndarray, target_len: int) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    if len(poses) == target_len:
        return poses
    if len(poses) == 0:
        raise ExportError("cannot resample an empty pose sequence")
    if len(poses) == 1:
        return np.repeat(poses[:1], target_len, axis=0)
    src_t = np.linspace(0.0, 1.0, len(poses))
    dst_t = np.linspace(0.0, 1.0, target_len)
    trans = np.stack([np.interp(dst_t, src_t, poses[:, i, 3]) for i in range(3)], axis=1)
    out = np.tile(np.eye(4, dtype=float), (target_len, 1, 1))
    quats = np.stack([matrix_to_quat_xyzw(orthonormalized(p[:3, :3])) for p in poses], axis=0)
    for out_idx, t in enumerate(dst_t):
        right = int(np.searchsorted(src_t, t, side="right"))
        if right <= 0:
            quat = quats[0]
        elif right >= len(src_t):
            quat = quats[-1]
        else:
            left = right - 1
            alpha = float((t - src_t[left]) / (src_t[right] - src_t[left]))
            quat = quat_slerp(quats[left], quats[right], alpha)
        out[out_idx, :3, :3] = quat_xyzw_to_matrix(quat)
    out[:, :3, 3] = trans
    return out


def inspire_action_to_qpos_dof6(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=float)
    qpos = np.zeros((action.shape[0], 6), dtype=float)
    qpos[:, 0] = 1.40 * (1.0 - action[:, 5] / 1000.0)
    qpos[:, 1] = 0.60 * (1.0 - action[:, 4] / 1000.0)
    for dst, src in ((2, 3), (3, 2), (4, 1), (5, 0)):
        qpos[:, dst] = (
            -4e-8 * action[:, src] ** 3
            + 3e-5 * action[:, src] ** 2
            - 0.0704 * action[:, src]
            + 83.572
        ) * np.pi / 180.0
    return qpos


def inspire_f1_action_to_qpos_dof6(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=float)
    qpos = np.zeros((action.shape[0], 6), dtype=float)
    qpos[:, 0] = (1800.0 - action[:, 0]) * np.pi / 1800.0
    qpos[:, 1] = (1350.0 - action[:, 1]) * np.pi / 1800.0
    qpos[:, 2] = (1740.0 - action[:, 2]) * np.pi / 1800.0
    qpos[:, 3] = (1740.0 - action[:, 3]) * np.pi / 1800.0
    qpos[:, 4] = (1740.0 - action[:, 4]) * np.pi / 1800.0
    qpos[:, 5] = (1740.0 - action[:, 5]) * np.pi / 1800.0
    return qpos


def load_robot_qpos(
    episode_root: Path,
    hand: str,
    arm_time_offset: float = 0.09,
    hand_time_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    arm_qpos, arm_time = load_series(
        episode_root / "raw" / "arm",
        ("position.npy", "action_qpos.npy", "action.npy"),
    )
    hand_dir = episode_root / "raw" / "hand"
    if hand == "inspire_f1":
        hand_raw, hand_time = load_series(hand_dir, ("right_joint_states.npy", "right_commands.npy"))
        hand_qpos = inspire_f1_action_to_qpos_dof6(hand_raw)
    elif hand in {"inspire", "inspire_new"}:
        hand_raw, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))
        hand_qpos = inspire_action_to_qpos_dof6(hand_raw)
    elif hand in {"allegro", "allegro_v5"}:
        hand_qpos, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))
    else:
        raise ExportError(f"unsupported robot hand: {hand}")
    if len(arm_qpos) == 0:
        if len(hand_qpos) == 0:
            raise ExportError(f"empty arm and hand trajectories: {episode_root}")
        arm_qpos = np.zeros((len(hand_qpos), 6), dtype=float)
        arm_time = hand_time
    if arm_time_offset != 0.0:
        arm_time = arm_time + float(arm_time_offset)
    if hand_time_offset != 0.0:
        hand_time = hand_time + float(hand_time_offset)
    hand_qpos = resample_to(hand_time, hand_qpos, arm_time)
    n = min(len(arm_qpos), len(hand_qpos), len(arm_time))
    return np.concatenate([arm_qpos[:n], hand_qpos[:n]], axis=1), arm_time[:n]


def load_video_timeline(episode_root: Path, fallback_len: int) -> tuple[np.ndarray, np.ndarray]:
    ts_dir = episode_root / "raw" / "timestamps"
    ts_path = ts_dir / "timestamp.npy"
    frame_id_path = ts_dir / "frame_id.npy"
    if ts_path.exists() and frame_id_path.exists():
        ts = load_array(ts_path).reshape(-1)
        frame_ids = np.asarray(np.load(frame_id_path), dtype=int).reshape(-1)
        n = min(len(ts), len(frame_ids))
        return ts[:n], frame_ids[:n]
    return np.arange(fallback_len, dtype=float), np.arange(1, fallback_len + 1, dtype=int)


def load_robot_qpos_on_video_timeline(
    episode_root: Path,
    hand: str,
    arm_time_offset: float = 0.09,
    hand_time_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_qpos, robot_time = load_robot_qpos(
        episode_root,
        hand,
        arm_time_offset=arm_time_offset,
        hand_time_offset=hand_time_offset,
    )
    video_time, frame_ids = load_video_timeline(episode_root, len(full_qpos))
    return resample_to(robot_time, full_qpos, video_time), video_time, frame_ids


POSE_DIR_CANDIDATES = (
    "object_6d",
    "object_tracking",
    "sam3_seed_tracking_output/poses",
    "foundationpose_only_video_tracking_output/poses",
    "sequence_refine_output/refined_world_poses",
    "sequence_refine_output/seed_world_poses",
)


def default_asset_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "assets",
        dataset_root.parent / "v0" / "assets",
        dataset_root / "v0" / "assets",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def default_pose_root(dataset_root: Path) -> Path:
    if dataset_root.name == "v0":
        return dataset_root.parent
    return dataset_root


def find_pose_dir(episode_roots: Sequence[Path]) -> Path | None:
    for root in episode_roots:
        for name in POSE_DIR_CANDIDATES:
            candidate = root / name
            if candidate.is_dir() and any(candidate.glob("pose_*.txt")):
                return candidate
    return None


def load_pose_txt_sequence(pose_dir: Path) -> np.ndarray:
    poses = []
    for path in sorted(pose_dir.glob("pose_*.txt")):
        arr = np.loadtxt(path, dtype=float)
        if arr.shape == (16,):
            arr = arr.reshape(4, 4)
        if arr.shape == (4, 4):
            poses.append(arr)
    if not poses:
        raise ExportError(f"no pose_*.txt files found in {pose_dir}")
    return np.stack(poses, axis=0)


def load_obj_t_frames_sequence(path: Path) -> np.ndarray:
    data = np.load(path)

    def frame_number(key: str) -> int:
        match = re.fullmatch(r"frame_(\d+)", key)
        return int(match.group(1)) if match else -1

    keys = sorted((key for key in data.files if frame_number(key) >= 0), key=frame_number)
    if not keys:
        raise ExportError(f"no frame_N entries found in {path}")
    poses = []
    for key in keys:
        arr = np.asarray(data[key], dtype=float)
        if arr.shape == (4, 4):
            poses.append(arr)
    if not poses:
        raise ExportError(f"no 4x4 frame_N poses found in {path}")
    return np.stack(poses, axis=0)


def find_obj_t_frames(episode_roots: Sequence[Path]) -> Path | None:
    for root in episode_roots:
        candidates = (
            root / "object_tracking_result" / "obj_T_frames.npz",
            root / "obj_T_frames.npz",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def load_object_poses_robot(
    episode_roots: Sequence[Path],
    target_len: int,
    object_pose_dir: Path | None = None,
) -> tuple[np.ndarray, Path]:
    pose_dir = object_pose_dir if object_pose_dir is not None else find_pose_dir(episode_roots)
    if pose_dir is not None:
        if not pose_dir.is_dir():
            raise ExportError(f"object pose directory not found: {pose_dir}")
        poses = load_pose_txt_sequence(pose_dir)
        pose_source = pose_dir
    else:
        npz_path = find_obj_t_frames(episode_roots)
        if npz_path is None:
            searched = ", ".join(str(path) for path in episode_roots)
            raise ExportError(f"object pose source not found in any of: {searched}")
        poses = load_obj_t_frames_sequence(npz_path)
        pose_source = npz_path

    c2r_path = episode_roots[0] / "C2R.npy"
    c2r = load_array(c2r_path) if c2r_path.exists() else np.eye(4, dtype=float)
    poses_world = resample_poses(poses, target_len)
    robot_from_world = np.linalg.inv(c2r)
    return np.einsum("ij,tjk->tik", robot_from_world, poses_world), pose_source


def load_human_mano_sequence(episode_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    candidates = (
        episode_root / "mano" / "mano",
        episode_root / "mano",
        episode_root / "hand" / "mano",
        episode_root / "hand" / "mano" / "mano",
    )
    mano_paths: list[Path] = []
    mano_dir: Path | None = None
    for candidate in candidates:
        if candidate.is_dir():
            mano_paths = sorted(candidate.glob("*.obj"))
            if mano_paths:
                mano_dir = candidate
                break
    if not mano_paths or mano_dir is None:
        joined = ", ".join(str(path) for path in candidates)
        raise ExportError(f"MANO OBJ files not found. Checked: {joined}")

    vertices = []
    faces: np.ndarray | None = None
    frame_ids = []
    for path in mano_paths:
        mesh = parse_obj(path)
        mesh_faces = np.asarray(mesh.indices, dtype=np.uint32)
        if faces is None:
            faces = mesh_faces
        elif mesh_faces.shape != faces.shape or not np.array_equal(mesh_faces, faces):
            raise ExportError(f"MANO topology differs at {path}")
        vertices.append(np.asarray(mesh.vertices, dtype=np.float32))
        try:
            frame_ids.append(int(path.stem))
        except ValueError as exc:
            raise ExportError(f"MANO OBJ filename is not a frame id: {path}") from exc
    if faces is None:
        raise ExportError(f"no MANO faces loaded from {mano_dir}")
    return np.stack(vertices, axis=0), faces, np.asarray(frame_ids, dtype=int), mano_dir


def load_root_from_world(episode_root: Path) -> np.ndarray:
    c2r_path = episode_root / "C2R.npy"
    c2r = load_array(c2r_path) if c2r_path.exists() else np.eye(4, dtype=float)
    return np.linalg.inv(c2r)


def parse_obj(path: Path) -> MeshData:
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    faces: list[list[int]] = []
    saw_color = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    saw_color = True
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                else:
                    colors.append([1.0, 1.0, 1.0])
            elif line.startswith("f "):
                raw = line.split()[1:]
                idxs = []
                for item in raw:
                    head = item.split("/")[0]
                    if not head:
                        continue
                    idx = int(head)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    idxs.append(idx)
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if not vertices or not faces:
        raise ExportError(f"OBJ has no triangle geometry: {path}")
    color_array = np.asarray(colors, dtype=np.float32) if saw_color else None
    return MeshData(
        name=path.stem,
        vertices=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(faces, dtype=np.uint32),
        colors=color_array,
    )


def parse_binary_stl(path: Path, data: bytes) -> MeshData:
    if len(data) < 84:
        raise ExportError(f"binary STL is too small: {path}")
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected > len(data):
        raise ExportError(f"binary STL is truncated: {path}")
    vertices = np.empty((tri_count * 3, 3), dtype=np.float32)
    indices = np.empty((tri_count, 3), dtype=np.uint32)
    offset = 84
    for tri_idx in range(tri_count):
        offset += 12
        coords = struct.unpack_from("<9f", data, offset)
        vertices[tri_idx * 3 : tri_idx * 3 + 3] = np.asarray(coords, dtype=np.float32).reshape(3, 3)
        indices[tri_idx] = (tri_idx * 3, tri_idx * 3 + 1, tri_idx * 3 + 2)
        offset += 38
    return MeshData(name=path.stem, vertices=vertices, indices=indices)


def parse_ascii_stl(path: Path, text: str) -> MeshData:
    vertices: list[list[float]] = []
    indices: list[list[int]] = []
    tri: list[int] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            tri.append(len(vertices) - 1)
            if len(tri) == 3:
                indices.append(tri)
                tri = []
    if not vertices or not indices:
        raise ExportError(f"ASCII STL has no triangle geometry: {path}")
    return MeshData(
        name=path.stem,
        vertices=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
    )


def load_mesh(path: Path) -> MeshData:
    path = path.resolve()
    cached = _MESH_CACHE.get(path)
    if cached is not None:
        return cached
    suffix = path.suffix.lower()
    if suffix == ".obj":
        mesh = parse_obj(path)
    elif suffix == ".stl":
        data = path.read_bytes()
        if len(data) >= 84:
            tri_count = struct.unpack_from("<I", data, 80)[0]
            if 84 + tri_count * 50 == len(data):
                mesh = parse_binary_stl(path, data)
            else:
                mesh = parse_ascii_stl(path, data.decode("utf-8", errors="ignore"))
        else:
            mesh = parse_ascii_stl(path, data.decode("utf-8", errors="ignore"))
    else:
        raise ExportError(f"unsupported mesh format: {path}")
    _MESH_CACHE[path] = mesh
    return mesh


def transformed_mesh(mesh: MeshData, transform: np.ndarray, scale: np.ndarray | None = None) -> MeshData:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if scale is not None:
        vertices = vertices * np.asarray(scale, dtype=np.float32).reshape(1, 3)
    vertices = apply_transform(vertices, transform).astype(np.float32)
    return MeshData(
        name=mesh.name,
        vertices=vertices,
        indices=mesh.indices,
        colors=mesh.colors,
        mode=mesh.mode,
    )


def decimation_indices(length: int, stride: int) -> np.ndarray:
    if stride <= 1 or length <= 2:
        return np.arange(length, dtype=np.int64)
    indices = np.arange(0, length, stride, dtype=np.int64)
    if indices[-1] != length - 1:
        indices = np.concatenate([indices, np.asarray([length - 1], dtype=np.int64)])
    return indices


def is_arm_visual(visual: VisualSpec) -> bool:
    return visual.link in {"link_base", "link1", "link2", "link3", "link4", "link5", "link6"}


def resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    filename = re.sub(r"^package://[^/]+/", "", filename)
    return (urdf_path.parent / filename).resolve()


def parse_urdf(urdf_path: Path) -> UrdfModel:
    urdf_path = urdf_path.resolve()
    cached = _URDF_CACHE.get(urdf_path)
    if cached is not None:
        return cached
    root = ET.parse(urdf_path).getroot()
    materials: dict[str, tuple[float, float, float, float]] = {}
    for mat in root.findall("material"):
        name = mat.attrib.get("name")
        color = mat.find("color")
        if name and color is not None and color.attrib.get("rgba"):
            rgba = tuple(float(x) for x in color.attrib["rgba"].split())
            if len(rgba) == 4:
                materials[name] = rgba  # type: ignore[assignment]

    parented: set[str] = set()
    joints: list[JointSpec] = []
    qpos_index = 0
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "fixed")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.attrib["link"]
        child = child_el.attrib["link"]
        parented.add(child)
        origin_el = joint.find("origin")
        xyz = parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, (0, 0, 0))
        rpy = parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, (0, 0, 0))
        axis_el = joint.find("axis")
        axis = parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, (0, 0, 1))
        mimic_el = joint.find("mimic")
        mimic_joint = mimic_el.attrib.get("joint") if mimic_el is not None else None
        mimic_multiplier = float(mimic_el.attrib.get("multiplier", "1.0")) if mimic_el is not None else 1.0
        mimic_offset = float(mimic_el.attrib.get("offset", "0.0")) if mimic_el is not None else 0.0
        idx = None
        if joint_type != "fixed" and mimic_joint is None:
            idx = qpos_index
            qpos_index += 1
        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=make_transform(xyz, rpy),
                axis=axis,
                qpos_index=idx,
                mimic_joint=mimic_joint,
                mimic_multiplier=mimic_multiplier,
                mimic_offset=mimic_offset,
            )
        )

    links = [link.attrib["name"] for link in root.findall("link")]
    root_link = next((link for link in links if link not in parented), links[0] if links else "world")
    visuals: list[VisualSpec] = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for visual in link.findall("visual"):
            mesh_el = visual.find("geometry/mesh")
            if mesh_el is None or not mesh_el.attrib.get("filename"):
                continue
            origin_el = visual.find("origin")
            xyz = parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, (0, 0, 0))
            rpy = parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, (0, 0, 0))
            scale = parse_vec(mesh_el.attrib.get("scale"), (1, 1, 1))
            material_el = visual.find("material")
            material = material_el.attrib.get("name") if material_el is not None else None
            visuals.append(
                VisualSpec(
                    link=link_name,
                    mesh_path=resolve_mesh_path(urdf_path, mesh_el.attrib["filename"]),
                    origin=make_transform(xyz, rpy),
                    scale=scale,
                    material=material,
                )
            )

    child_joints: dict[str, list[JointSpec]] = {}
    for joint in joints:
        child_joints.setdefault(joint.parent, []).append(joint)
    model = UrdfModel(root_link=root_link, joints=joints, visuals=visuals, materials=materials, child_joints=child_joints)
    _URDF_CACHE[urdf_path] = model
    return model


def compute_link_transforms(urdf: UrdfModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    qpos = np.asarray(qpos, dtype=float).reshape(-1)
    transforms: dict[str, np.ndarray] = {urdf.root_link: np.eye(4, dtype=float)}
    joint_values: dict[str, float] = {}
    stack = [urdf.root_link]
    while stack:
        parent = stack.pop()
        parent_tf = transforms[parent]
        for joint in urdf.child_joints.get(parent, []):
            value = 0.0
            if joint.mimic_joint is not None:
                value = joint_values.get(joint.mimic_joint, 0.0) * joint.mimic_multiplier + joint.mimic_offset
            elif joint.qpos_index is not None and joint.qpos_index < len(qpos):
                value = float(qpos[joint.qpos_index])
            joint_values[joint.name] = value
            transforms[joint.child] = parent_tf @ joint.origin @ axis_angle_transform(joint.axis, value, joint.joint_type)
            stack.append(joint.child)
    return transforms
