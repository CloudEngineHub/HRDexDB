# HRDexDB Contact Heatmaps

This folder contains self-contained utilities for generating and visualizing
per-vertex contact/proximity heatmaps from HRDexDB MANO meshes, robot URDFs,
object meshes, object poses, and robot or human hand trajectories.

The code does not import from the rest of the HRDexDB repository. Keep
`generate_contact_heatmaps.py`, `hrdexdb_io.py`, and `visualize_heatmap.py`
together when releasing this folder.

## Inputs

- HRDexDB episode root under your local dataset path, for example
  `"<put the dataset path>/inspire_f1/apple/0"`.
- HRDexDB assets under the dataset `v0/assets` folder.
- Object meshes from `"<put the dataset path to mesh_blender>/<object>/<object>_viser.obj"`
  by default.
- Object pose folders such as `sequence_refine_output/refined_world_poses`.
- MANO meshes for human episodes or robot qpos plus URDFs for robot episodes.

## Generate Heatmaps

Example for the 326th frame of `inspire_f1/apple/0`:

```bash
# Replace "" with your paths. Put the dataset path in --dataset-root.
python hrdexdb_contact_heatmaps/generate_contact_heatmaps.py \
  --dataset-root "" \
  --mesh "" \
  --hand inspire_f1 \
  --object apple \
  --scene 0 \
  --frame-id 326 \
  --output hrdexdb_contact_heatmaps/output/inspire_f1/apple/0/frame_326/contact_heatmaps.npz \
  --summary-json hrdexdb_contact_heatmaps/output/inspire_f1/apple/0/frame_326/summary.json
```

If `--output` is omitted, the default output path is:

```text
hrdexdb_contact_heatmaps/output/<hand>/<object>/<scene>/contact_heatmaps.npz
```

Useful options:

- `--frame-id`: select a single HRDexDB frame id.
- `--frame-index`: select a single zero-based timeline index.
- `--frame-stride`: sample a full episode at a fixed stride.
- `--max-frames`: cap the number of sampled frames.
- `--distance-clip`: distance threshold in meters, default `0.02`.
- `--robot-visual-scope hand`: use hand/finger visual meshes only.
- `--robot-visual-scope all`: include arm visual meshes too.

## NPZ Format

The output is a compressed NumPy `.npz` file. Main arrays:

- `effector_distance`: `(T, N_effector)` float32, nearest object vertex distance in meters, clipped.
- `object_distance`: `(T, N_object)` float32, nearest effector vertex distance in meters, clipped.
- `effector_heat`: `(T, N_effector)` float32, `1 - clip(distance / distance_clip_m, 0, 1)`.
- `object_heat`: `(T, N_object)` float32, same heat formula for object vertices.
- `frame_ids`: `(T,)` int64 HRDexDB frame ids.
- `sample_indices`: `(T,)` int64 zero-based timeline indices.
- `object_vertices_local`: `(N_object, 3)` float32 object mesh vertices in local coordinates.
- `object_vertices_first_root`: `(N_object, 3)` float32 object vertices posed in HRDexDB root coordinates for the first sampled frame.
- `object_faces`: `(F_object, 3)` uint32 triangle indices.
- `effector_vertices_first_root`: `(N_effector, 3)` float32 MANO or robot vertices in HRDexDB root coordinates for the first sampled frame.
- `effector_faces`: `(F_effector, 3)` uint32 triangle indices.
- `effector_vertex_link_index`, `effector_vertex_offsets`, `effector_link_names`: robot-link bookkeeping.
- `metadata_json`: JSON string with episode path, pose source, mesh source, selected frames, and shape metadata.

## Visualize Heatmaps

Export colored PLY meshes:

```bash
python hrdexdb_contact_heatmaps/visualize_heatmap.py \
  hrdexdb_contact_heatmaps/output/inspire_f1/apple/0/frame_326/contact_heatmaps.npz \
  --output-root hrdexdb_contact_heatmaps/output/inspire_f1/apple/0/frame_326/vis
```

By default this writes:

- `object_heat_frame000.ply`
- `effector_heat_frame000.ply`
- `combined_heat_frame000.ply`
- `manifest.json`

PLY files are `binary_little_endian` meshes with per-vertex `red`, `green`,
`blue`, and `alpha` properties. The color ramp matches the Paradex-style
blue-green-red mapping: heat `0` is blue and heat `1` is red.

## Grasp-Moment Annotations

`grasp_moment_annotations.json` contains normalized grasping-moment frame
annotations for successful episodes:

- `annotations.allegro_v5`: object -> episode -> grasp-frame record.
- `annotations.human`: object -> episode -> success-filtered human grasp-frame record.
- `annotations.inspire_f1`: object -> episode -> success-filtered inspire grasp-frame record.
- `paired_successes`: pair id -> human/inspire pair record, preserving the original successful pairing.

The file was built from internal annotation sources before public release:

- `""` for `allegro_v5`. Put the dataset path if rebuilding from source annotations.
- `""` for `human` and `inspire_f1`. Put the dataset path if rebuilding from source annotations.

Summary in the generated JSON:

- `allegro_v5_annotated_episodes`: 487
- `paired_source_total_pairs`: 512
- `paired_source_success_pairs`: 421
- `paired_source_fail_pairs_excluded`: 91
- `human_successful_episodes`: 421
- `inspire_f1_successful_episode_keys`: 419
- `inspire_f1_successful_pairs`: 421
- `inspire_f1_episode_keys_with_multiple_success_pairs`: 2

The paired source has explicit `status == "success"` labels, and only those
records are included. The Allegro source has no explicit status/failure field,
so all records in that file are included as annotated successful episodes.

## Dependencies

Required:

- Python 3.10+
- NumPy

Optional but recommended:

- SciPy, for faster nearest-neighbor queries through `scipy.spatial.cKDTree`.

If SciPy is unavailable, the generator falls back to a bounded voxel search.
