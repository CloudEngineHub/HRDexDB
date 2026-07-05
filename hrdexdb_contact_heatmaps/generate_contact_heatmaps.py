#!/usr/bin/env python3
"""Generate per-vertex hand/object proximity heatmaps for one HRDexDB episode."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hrdexdb_io import (
    ROBOT_URDFS,
    ExportError,
    MeshData,
    apply_transform,
    compute_link_transforms,
    decimation_indices,
    default_asset_root,
    default_pose_root,
    is_arm_visual,
    load_human_mano_sequence,
    load_mesh,
    load_object_poses_robot,
    load_robot_qpos_on_video_timeline,
    load_root_from_world,
    parse_urdf,
    transformed_mesh,
)

DEFAULT_MESH_BLENDER_ROOT = ""  # put the dataset path to the mesh_blender folder


@dataclass(frozen=True)
class EffectorMesh:
    vertices: np.ndarray
    faces: np.ndarray
    link_names: list[str]
    vertex_link_index: np.ndarray
    vertex_offsets: np.ndarray


_CKDTREE: Any | None | bool = None


def _get_ckdtree() -> Any | None:
    global _CKDTREE
    if _CKDTREE is False:
        return None
    if _CKDTREE is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from scipy.spatial import cKDTree

            _CKDTREE = cKDTree
        except Exception:
            _CKDTREE = False
            return None
    return _CKDTREE


def _cell_groups(points: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.floor(points / float(cell_size)).astype(np.int64)
    order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    if len(sorted_cells) == 0:
        return cells, order, np.asarray([], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_cells, axis=0) != 0, axis=1)) + 1]
    return sorted_cells, order, starts


def clipped_nearest_distances_voxel(points: np.ndarray, targets: np.ndarray, max_distance: float) -> np.ndarray:
    max_distance = max(float(max_distance), 1e-8)
    out = np.full((len(points),), max_distance, dtype=np.float32)

    target_cells, target_order, target_starts = _cell_groups(targets, max_distance)
    target_grid: dict[tuple[int, int, int], np.ndarray] = {}
    for pos, start in enumerate(target_starts):
        stop = int(target_starts[pos + 1]) if pos + 1 < len(target_starts) else len(target_order)
        key = tuple(int(x) for x in target_cells[int(start)])
        target_grid[key] = target_order[int(start) : stop]

    query_cells, query_order, query_starts = _cell_groups(points, max_distance)
    neighbor_offsets = list(itertools.product((-1, 0, 1), repeat=3))
    max_sq = np.float32(max_distance * max_distance)
    chunk = 512

    for pos, start in enumerate(query_starts):
        stop = int(query_starts[pos + 1]) if pos + 1 < len(query_starts) else len(query_order)
        cell = query_cells[int(start)]
        candidate_blocks = []
        for offset in neighbor_offsets:
            key = (int(cell[0] + offset[0]), int(cell[1] + offset[1]), int(cell[2] + offset[2]))
            block = target_grid.get(key)
            if block is not None:
                candidate_blocks.append(block)
        if not candidate_blocks:
            continue
        candidate_idx = np.concatenate(candidate_blocks)
        candidate_points = targets[candidate_idx]
        query_idx = query_order[int(start) : stop]
        for chunk_start in range(0, len(query_idx), chunk):
            idx = query_idx[chunk_start : chunk_start + chunk]
            query = points[idx]
            dist2 = np.sum((query[:, None, :] - candidate_points[None, :, :]) ** 2, axis=2)
            best = np.min(dist2, axis=1)
            near = best < max_sq
            if np.any(near):
                vals = np.sqrt(best[near]).astype(np.float32)
                out[idx[near]] = vals
    return out


def nearest_distances(points: np.ndarray, targets: np.ndarray, max_distance: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"targets must have shape (M, 3), got {targets.shape}")
    if len(points) == 0 or len(targets) == 0:
        return np.full((len(points),), np.inf, dtype=np.float32)
    cKDTree = _get_ckdtree()
    if cKDTree is not None:
        distances = cKDTree(targets).query(points, k=1, workers=-1)[0].astype(np.float32)
        return np.minimum(distances, np.float32(max_distance))
    return clipped_nearest_distances_voxel(points, targets, max_distance)


def distances_to_heat(distances: np.ndarray, clip_distance: float) -> np.ndarray:
    clip = max(float(clip_distance), 1e-8)
    return (1.0 - np.clip(np.asarray(distances, dtype=np.float32) / clip, 0.0, 1.0)).astype(np.float32)


def resolve_object_mesh(
    *,
    asset_root: Path,
    mesh_blender_root: Path,
    object_name: str,
    object_mesh_path: Path | None,
    object_mesh_source: str,
) -> tuple[Path, str]:
    if object_mesh_path is not None:
        path = object_mesh_path.expanduser().resolve()
        if not path.is_file():
            raise ExportError(f"object mesh not found: {path}")
        return path, "explicit"

    viser_path = mesh_blender_root / object_name / f"{object_name}_viser.obj"
    asset_path = asset_root / "mesh" / object_name / f"{object_name}.obj"
    candidates = (viser_path, asset_path) if object_mesh_source == "viser" else (asset_path, viser_path)
    for path in candidates:
        if path.is_file():
            return path.resolve(), "viser" if path == viser_path else "asset"
    raise ExportError(f"object mesh not found; tried {viser_path} and {asset_path}")


def robot_effector_mesh_at_qpos(
    urdf_path: Path,
    qpos: np.ndarray,
    visual_scope: str,
    local_mesh_cache: dict[tuple[Path, tuple[float, float, float], bytes], MeshData],
) -> EffectorMesh:
    urdf = parse_urdf(urdf_path)
    link_tfs = compute_link_transforms(urdf, qpos)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    link_names: list[str] = []
    vertex_link_index: list[np.ndarray] = []
    offsets = [0]

    for visual in urdf.visuals:
        if visual_scope == "hand" and is_arm_visual(visual):
            continue
        if not visual.mesh_path.is_file():
            raise ExportError(f"URDF visual mesh missing: {visual.mesh_path}")
        key = (visual.mesh_path.resolve(), tuple(float(x) for x in visual.scale), visual.origin.tobytes())
        if key not in local_mesh_cache:
            local_mesh_cache[key] = transformed_mesh(load_mesh(visual.mesh_path), visual.origin, visual.scale)
        local_mesh = local_mesh_cache[key]
        link_vertices = apply_transform(local_mesh.vertices, link_tfs.get(visual.link, np.eye(4, dtype=float)))
        vertices.append(np.asarray(link_vertices, dtype=np.float32))
        faces.append(np.asarray(local_mesh.indices, dtype=np.uint32) + offsets[-1])
        link_names.append(visual.link)
        vertex_link_index.append(np.full((len(local_mesh.vertices),), len(link_names) - 1, dtype=np.int32))
        offsets.append(offsets[-1] + len(local_mesh.vertices))

    if not vertices:
        raise ExportError(f"no URDF visual meshes selected from {urdf_path} with scope={visual_scope}")
    return EffectorMesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        link_names=link_names,
        vertex_link_index=np.concatenate(vertex_link_index, axis=0),
        vertex_offsets=np.asarray(offsets, dtype=np.int64),
    )


def build_episode_heatmaps(
    *,
    dataset_root: Path,
    pose_root: Path,
    asset_root: Path,
    mesh_blender_root: Path,
    hand: str,
    object_name: str,
    scene: str,
    output_path: Path,
    object_mesh_path: Path | None,
    object_mesh_source: str,
    object_pose_dir: Path | None,
    frame_stride: int,
    max_frames: int | None,
    frame_id: int | None,
    frame_index: int | None,
    distance_clip: float,
    robot_visual_scope: str,
    arm_time_offset: float,
    hand_time_offset: float,
    human_fps: float,
) -> dict[str, Any]:
    episode_root = dataset_root / hand / object_name / scene
    if not episode_root.is_dir():
        raise ExportError(f"episode directory not found: {episode_root}")

    object_mesh_file, object_mesh_source_used = resolve_object_mesh(
        asset_root=asset_root,
        mesh_blender_root=mesh_blender_root,
        object_name=object_name,
        object_mesh_path=object_mesh_path,
        object_mesh_source=object_mesh_source,
    )
    object_mesh = load_mesh(object_mesh_file)
    object_vertices_local = np.asarray(object_mesh.vertices, dtype=np.float32)

    robot_urdf_path = None
    local_robot_mesh_cache: dict[tuple[Path, tuple[float, float, float], bytes], MeshData] = {}
    if hand == "human":
        effector_vertices_seq, effector_faces, frame_ids, mano_dir = load_human_mano_sequence(episode_root)
        timeline_len = len(effector_vertices_seq)
        root_from_world = load_root_from_world(episode_root)
        effector_kind = "mano"
        effector_source = str(mano_dir)
        link_names: list[str] = ["human_mano"]
        vertex_link_index = np.zeros((effector_vertices_seq.shape[1],), dtype=np.int32)
        vertex_offsets = np.asarray([0, effector_vertices_seq.shape[1]], dtype=np.int64)
        if human_fps <= 0:
            raise ExportError(f"human_fps must be positive, got {human_fps}")
    else:
        robot_rel = ROBOT_URDFS.get(hand)
        if robot_rel is None:
            raise ExportError(f"no robot URDF mapping for hand={hand!r}")
        robot_urdf_path = asset_root / "robots" / robot_rel
        if not robot_urdf_path.is_file():
            raise ExportError(f"robot URDF not found: {robot_urdf_path}")
        qpos, _video_time, frame_ids = load_robot_qpos_on_video_timeline(
            episode_root,
            hand,
            arm_time_offset=arm_time_offset,
            hand_time_offset=hand_time_offset,
        )
        timeline_len = len(qpos)
        first_mesh = robot_effector_mesh_at_qpos(
            robot_urdf_path,
            qpos[0],
            robot_visual_scope,
            local_robot_mesh_cache,
        )
        effector_faces = first_mesh.faces
        link_names = first_mesh.link_names
        vertex_link_index = first_mesh.vertex_link_index
        vertex_offsets = first_mesh.vertex_offsets
        effector_kind = "robot_urdf"
        effector_source = str(robot_urdf_path)

    pose_episode_roots = [episode_root]
    fallback_pose_episode = pose_root / hand / object_name / scene
    if fallback_pose_episode != episode_root:
        pose_episode_roots.append(fallback_pose_episode)
    object_poses, pose_dir = load_object_poses_robot(
        pose_episode_roots,
        timeline_len,
        object_pose_dir=object_pose_dir,
    )

    if frame_id is not None and frame_index is not None:
        raise ExportError("use only one of frame_id or frame_index")
    if frame_id is not None:
        matches = np.flatnonzero(np.asarray(frame_ids, dtype=int) == int(frame_id))
        if len(matches) == 0:
            raise ExportError(f"frame_id {frame_id} not found in episode frame_ids")
        sample_idx = matches[:1].astype(np.int64)
        selection = f"frame_id:{frame_id}"
    elif frame_index is not None:
        if frame_index < 0 or frame_index >= timeline_len:
            raise ExportError(f"frame_index {frame_index} out of range [0, {timeline_len - 1}]")
        sample_idx = np.asarray([int(frame_index)], dtype=np.int64)
        selection = f"frame_index:{frame_index}"
    else:
        sample_idx = decimation_indices(timeline_len, max(1, int(frame_stride)))
        if max_frames is not None:
            sample_idx = sample_idx[: max(0, int(max_frames))]
        selection = f"stride:{frame_stride}"
    if len(sample_idx) == 0:
        raise ExportError("no frames selected")

    effector_distances: list[np.ndarray] = []
    object_distances: list[np.ndarray] = []
    first_effector_vertices: np.ndarray | None = None
    first_object_vertices: np.ndarray | None = None

    for idx in sample_idx:
        object_vertices = apply_transform(object_vertices_local, object_poses[int(idx)])
        if hand == "human":
            effector_vertices = apply_transform(effector_vertices_seq[int(idx)], root_from_world)
        else:
            effector_vertices = robot_effector_mesh_at_qpos(
                robot_urdf_path,
                qpos[int(idx)],
                robot_visual_scope,
                local_robot_mesh_cache,
            ).vertices
        if first_effector_vertices is None:
            first_effector_vertices = np.asarray(effector_vertices, dtype=np.float32)
        if first_object_vertices is None:
            first_object_vertices = np.asarray(object_vertices, dtype=np.float32)
        effector_distances.append(nearest_distances(effector_vertices, object_vertices, distance_clip))
        object_distances.append(nearest_distances(object_vertices, effector_vertices, distance_clip))

    effector_distance = np.stack(effector_distances, axis=0).astype(np.float32)
    object_distance = np.stack(object_distances, axis=0).astype(np.float32)
    effector_heat = distances_to_heat(effector_distance, distance_clip)
    object_heat = distances_to_heat(object_distance, distance_clip)

    metadata: dict[str, Any] = {
        "episode_root": str(episode_root),
        "hand": hand,
        "object": object_name,
        "scene": scene,
        "frames_total": int(timeline_len),
        "frames_sampled": int(len(sample_idx)),
        "frame_stride": int(frame_stride),
        "frame_selection": selection,
        "frame_ids": [int(frame_ids[int(sample_idx[0])]), int(frame_ids[int(sample_idx[-1])])],
        "sample_indices": [int(i) for i in sample_idx],
        "pose_dir": str(pose_dir),
        "object_mesh": str(object_mesh_file),
        "object_mesh_source": object_mesh_source_used,
        "effector_kind": effector_kind,
        "effector_source": effector_source,
        "robot_visual_scope": robot_visual_scope if hand != "human" else None,
        "distance_clip_m": float(distance_clip),
        "distance_kind": "unsigned nearest-vertex Euclidean distance in HRDexDB root coordinates, clipped at distance_clip_m",
        "heat_formula": "1 - clip(distance / distance_clip_m, 0, 1)",
        "arrays": {
            "effector_distance": list(effector_distance.shape),
            "object_distance": list(object_distance.shape),
            "effector_heat": list(effector_heat.shape),
            "object_heat": list(object_heat.shape),
            "object_vertices_first_root": list(first_object_vertices.shape),
            "effector_vertices_first_root": list(first_effector_vertices.shape),
        },
        "output": str(output_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        effector_distance=effector_distance,
        object_distance=object_distance,
        effector_heat=effector_heat,
        object_heat=object_heat,
        frame_ids=np.asarray([int(frame_ids[int(i)]) for i in sample_idx], dtype=np.int64),
        sample_indices=np.asarray(sample_idx, dtype=np.int64),
        object_vertices_local=object_vertices_local.astype(np.float32),
        object_vertices_first_root=np.asarray(first_object_vertices, dtype=np.float32),
        object_faces=np.asarray(object_mesh.indices, dtype=np.uint32),
        effector_vertices_first_root=np.asarray(first_effector_vertices, dtype=np.float32),
        effector_faces=np.asarray(effector_faces, dtype=np.uint32),
        effector_vertex_link_index=vertex_link_index,
        effector_vertex_offsets=vertex_offsets,
        effector_link_names=np.asarray(link_names, dtype=str),
        metadata_json=json.dumps(metadata, indent=2, sort_keys=True),
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None, help="required; put the dataset path")
    parser.add_argument("--pose-root", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument(
        "--mesh-blender-root",
        type=Path,
        default=None,
        help="required unless --object-mesh-path is set; put the dataset path to mesh_blender",
    )
    parser.add_argument("--hand", default="human")
    parser.add_argument("--object", dest="object_name", default="apple")
    parser.add_argument("--scene", default="0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--object-mesh-path", type=Path, default=None)
    parser.add_argument("--object-mesh-source", choices=("viser", "asset"), default="viser")
    parser.add_argument("--object-pose-dir", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-id", type=int, default=None, help="select one frame by HRDexDB frame_id value")
    parser.add_argument("--frame-index", type=int, default=None, help="select one zero-based timeline index")
    parser.add_argument("--distance-clip", type=float, default=0.02)
    parser.add_argument("--robot-visual-scope", choices=("hand", "all"), default="hand")
    parser.add_argument("--arm-time-offset", type=float, default=0.09)
    parser.add_argument("--hand-time-offset", type=float, default=0.0)
    parser.add_argument("--human-fps", type=float, default=30.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def required_path(value: str | Path, option_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ExportError(f"{option_name} is required; put the dataset path")
    return Path(value).expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset_root = required_path(args.dataset_root, "--dataset-root")
        mesh_blender_root = (
            required_path(args.mesh_blender_root, "--mesh-blender-root")
            if args.object_mesh_path is None
            else Path("")
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1
    pose_root = args.pose_root.expanduser().resolve() if args.pose_root else default_pose_root(dataset_root).resolve()
    asset_root = args.asset_root.expanduser().resolve() if args.asset_root else default_asset_root(dataset_root).resolve()
    output_path = args.output
    if output_path is None:
        output_path = (
            Path("hrdexdb_contact_heatmaps")
            / "output"
            / args.hand
            / args.object_name
            / args.scene
            / "contact_heatmaps.npz"
        )
    try:
        metadata = build_episode_heatmaps(
            dataset_root=dataset_root,
            pose_root=pose_root,
            asset_root=asset_root,
            mesh_blender_root=mesh_blender_root,
            hand=args.hand,
            object_name=args.object_name,
            scene=args.scene,
            output_path=output_path.expanduser().resolve(),
            object_mesh_path=args.object_mesh_path,
            object_mesh_source=args.object_mesh_source,
            object_pose_dir=args.object_pose_dir.expanduser().resolve() if args.object_pose_dir else None,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            frame_id=args.frame_id,
            frame_index=args.frame_index,
            distance_clip=args.distance_clip,
            robot_visual_scope=args.robot_visual_scope,
            arm_time_offset=args.arm_time_offset,
            hand_time_offset=args.hand_time_offset,
            human_fps=args.human_fps,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
