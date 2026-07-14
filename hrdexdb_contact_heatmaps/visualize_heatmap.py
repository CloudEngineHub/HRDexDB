#!/usr/bin/env python3
"""Export HRDexDB contact heatmaps as colored PLY meshes."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def heat_to_rgba(heat: np.ndarray) -> np.ndarray:
    """Map heat values to the same blue-green-red ramp used by Paradex tools."""
    x = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    alpha = np.full((x.shape[0], 1), 255, dtype=np.uint8)
    rgb = np.stack([red, green, blue], axis=1) * 255.0
    return np.concatenate([rgb.astype(np.uint8), alpha], axis=1)


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    colors = np.asarray(colors, dtype=np.uint8)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if colors.shape != (vertices.shape[0], 4):
        raise ValueError(f"colors must have shape ({vertices.shape[0]}, 4), got {colors.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ]
    )
    vertex_records = np.empty(vertices.shape[0], dtype=vertex_dtype)
    vertex_records["x"] = vertices[:, 0]
    vertex_records["y"] = vertices[:, 1]
    vertex_records["z"] = vertices[:, 2]
    vertex_records["red"] = colors[:, 0]
    vertex_records["green"] = colors[:, 1]
    vertex_records["blue"] = colors[:, 2]
    vertex_records["alpha"] = colors[:, 3]

    with path.open("wb") as f:
        f.write(header)
        vertex_records.tofile(f)
        for tri in faces:
            f.write(struct.pack("<BIII", 3, int(tri[0]), int(tri[1]), int(tri[2])))


def require_array(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in data:
        raise KeyError(f"{name!r} missing from input NPZ")
    return np.asarray(data[name])


def stats(heat: np.ndarray) -> dict[str, Any]:
    heat = np.asarray(heat, dtype=np.float32)
    return {
        "min": float(np.min(heat)),
        "max": float(np.max(heat)),
        "mean": float(np.mean(heat)),
        "nonzero_vertices": int(np.count_nonzero(heat > 0.0)),
        "vertices": int(heat.shape[0]),
    }


def export_heatmaps(input_path: Path, output_root: Path, frame: int, mesh: str) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    outputs: dict[str, str] = {}
    heat_stats: dict[str, Any] = {}
    with np.load(input_path, allow_pickle=False) as data:
        object_heat_seq = require_array(data, "object_heat")
        effector_heat_seq = require_array(data, "effector_heat")
        if frame < 0 or frame >= object_heat_seq.shape[0]:
            raise IndexError(f"frame {frame} out of range [0, {object_heat_seq.shape[0] - 1}]")
        if frame >= effector_heat_seq.shape[0]:
            raise IndexError(f"frame {frame} out of range [0, {effector_heat_seq.shape[0] - 1}]")

        if "object_vertices_first_root" in data:
            object_vertices = np.asarray(data["object_vertices_first_root"], dtype=np.float32)
        else:
            object_vertices = np.asarray(data["object_vertices_local"], dtype=np.float32)
        object_faces = np.asarray(require_array(data, "object_faces"), dtype=np.uint32)
        object_colors = heat_to_rgba(object_heat_seq[frame])

        effector_vertices = np.asarray(require_array(data, "effector_vertices_first_root"), dtype=np.float32)
        effector_faces = np.asarray(require_array(data, "effector_faces"), dtype=np.uint32)
        effector_colors = heat_to_rgba(effector_heat_seq[frame])

        if mesh in {"object", "all"}:
            path = output_root / f"object_heat_frame{frame:03d}.ply"
            write_binary_ply(path, object_vertices, object_faces, object_colors)
            outputs["object"] = str(path)
            heat_stats["object"] = stats(object_heat_seq[frame])

        if mesh in {"effector", "all"}:
            path = output_root / f"effector_heat_frame{frame:03d}.ply"
            write_binary_ply(path, effector_vertices, effector_faces, effector_colors)
            outputs["effector"] = str(path)
            heat_stats["effector"] = stats(effector_heat_seq[frame])

        if mesh in {"combined", "all"}:
            path = output_root / f"combined_heat_frame{frame:03d}.ply"
            vertices = np.concatenate([object_vertices, effector_vertices], axis=0)
            faces = np.concatenate([object_faces, effector_faces + object_vertices.shape[0]], axis=0)
            colors = np.concatenate([object_colors, effector_colors], axis=0)
            write_binary_ply(path, vertices, faces, colors)
            outputs["combined"] = str(path)
            heat_stats["combined"] = {
                "vertices": int(vertices.shape[0]),
                "faces": int(faces.shape[0]),
            }

        frame_ids = data["frame_ids"].astype(np.int64).tolist() if "frame_ids" in data else None
        sample_indices = data["sample_indices"].astype(np.int64).tolist() if "sample_indices" in data else None
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data else None

    manifest = {
        "input": str(input_path),
        "frame": int(frame),
        "frame_id": int(frame_ids[frame]) if frame_ids is not None else None,
        "sample_index": int(sample_indices[frame]) if sample_indices is not None else None,
        "mesh": mesh,
        "outputs": outputs,
        "heat_stats": heat_stats,
        "input_metadata_json": metadata_json,
        "format": "binary_little_endian PLY with per-vertex RGBA uchar colors",
        "color_map": "blue-green-red ramp; heat 0 is blue, heat 1 is red",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="contact_heatmaps.npz file")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=0, help="zero-based sampled frame index in the NPZ")
    parser.add_argument(
        "--mesh",
        choices=("object", "effector", "combined", "all"),
        default="all",
        help="which colored mesh to export",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root
    if output_root is None:
        output_root = args.input.expanduser().resolve().parent / "vis"
    try:
        manifest = export_heatmaps(args.input, output_root, args.frame, args.mesh)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
