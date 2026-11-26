"""JSON response builders for frames, views and forward outputs."""

from __future__ import annotations

from typing import Any

from .core import Frame, View


def frame_summary(frame: Frame, initial_view_id: int | None = None) -> dict[str, Any]:
    root_id = initial_view_id
    if root_id is None:
        root_id = _root_view_id(frame)
    return {
        "frame_id": frame.frame_id,
        "id": frame.frame_id,
        "point_count": frame.num_points,
        "num_points": frame.num_points,
        "voxel_count": frame.num_voxels,
        "num_voxels": frame.num_voxels,
        "channels": frame.channels,
        "feature_channels": frame.channels,
        "coord_dim": frame.coord_dim,
        "initial_view_id": root_id,
        "initial_view": str(root_id),
        "created_at": frame.created_at,
    }


def _root_view_id(frame: Frame) -> int | None:
    # Late import to avoid a cycle; the store attaches views.
    from .core import store

    for view in store.views.values():
        if view.frame is frame and view.parent is None:
            return view.view_id
    return None


def view_meta(view: View) -> dict[str, Any]:
    return {
        "view_id": view.view_id,
        "id": view.view_id,
        "view_no": view.view_id,
        "view_ref_aliases": [str(view.view_id), f"v{view.view_id}", f"view-{view.view_id}"],
        "frame_id": view.frame.frame_id,
        "parent_view_id": view.parent.view_id if view.parent else None,
        "base_view_id": view.parent.view_id if view.parent else None,
        "depth": view.depth,
        "is_initial": view.parent is None,
        "note": view.note,
        "created_at": view.created_at,
        "changed_points": view.changed_points,
        "changed_point_count": view.changed_points,
        "changed_point_ids": [view.frame.point_ids[p] for p in view.changed_pidx],
        "recomputed_voxels": view.recomputed_voxels,
        "recomputed_voxel_count": view.recomputed_voxels,
        "recomputed_channels": view.recomputed_channels,
        "dirty_voxel_indices": [int(v) for v in view.dirty_vidx],
        "delta": view.delta_stats(),
        "resident_delta": view.delta_stats(),
    }


def view_output(view: View) -> dict[str, Any]:
    """Full forward output of a view (all currently non-empty voxels)."""
    frame = view.frame
    states = view.resolved_voxel_states()

    voxels: list[dict[str, Any]] = []
    voxel_coords: list[list[int]] = []
    max_values_col: list[list[float]] = []
    winner_col: list[list[Any]] = []
    source_col: list[int] = []
    empty_coords: list[list[int]] = []
    empty_indices: list[int] = []

    for vidx in sorted(states.keys()):
        values, winners_pidx, source_view = states[vidx]
        coord = frame.voxel_coord(vidx)
        if winners_pidx[0] < 0:
            empty_indices.append(vidx)
            empty_coords.append(coord)
            continue
        vals = [float(x) for x in values]
        winner_ids = [
            frame.point_ids[int(p)] if p >= 0 else None for p in winners_pidx
        ]
        entry = {
            "voxel_index": vidx,
            "voxel_id": vidx,
            "voxel_coord": coord,
            "coord": coord,
            "coordinates": coord,
            "is_empty": False,
            "nonempty": True,
            "max_values": vals,
            "values": vals,
            "channel_max": vals,
            "winner_point_ids": winner_ids,
            "argmax_point_ids": winner_ids,
            "winners": winner_ids,
            "source_view_id": source_view,
        }
        voxels.append(entry)
        voxel_coords.append(coord)
        max_values_col.append(vals)
        winner_col.append(winner_ids)
        source_col.append(source_view)

    meta = view_meta(view)
    meta.update(
        {
            "channels": frame.channels,
            "feature_channels": frame.channels,
            "nonempty_voxel_count": len(voxels),
            "non_empty_voxels": len(voxels),
            "voxel_count": frame.num_voxels,
            "voxels": voxels,
            # Columnar aliases, handy for vectorized training loops.
            "voxel_coords": voxel_coords,
            "coords": voxel_coords,
            "max_values": max_values_col,
            "values": max_values_col,
            "pooled_values": max_values_col,
            "winner_point_ids": winner_col,
            "argmax_point_ids": winner_col,
            "winner_source_view_ids": source_col,
            "empty_voxel_indices": empty_indices,
            "empty_voxel_coords": empty_coords,
        }
    )
    return meta


def frame_detail(frame: Frame) -> dict[str, Any]:
    summary = frame_summary(frame)
    voxels = []
    for vidx in range(frame.num_voxels):
        members = frame.members(vidx)
        voxels.append(
            {
                "voxel_index": vidx,
                "voxel_coord": frame.voxel_coord(vidx),
                "point_ids": [frame.point_ids[int(p)] for p in members],
                "point_count": int(len(members)),
            }
        )
    summary.update(
        {
            "point_ids": list(frame.point_ids),
            "voxels": voxels,
        }
    )
    return summary
