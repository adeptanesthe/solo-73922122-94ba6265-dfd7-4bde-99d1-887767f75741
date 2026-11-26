"""Tolerant parsing for the HTTP layer.

No external request schema was mandated, so these helpers accept the common
shapes a training script might use (columnar arrays, row objects, bare
lists, several naming conventions).  Malformed data raises
:class:`app.core.APIError` with an actionable 4xx code.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from .core import APIError, Frame, View


# --------------------------------------------------------------------------- #
# Small key helpers
# --------------------------------------------------------------------------- #
def _first_key(d: dict, keys: tuple[str, ...]) -> tuple[bool, Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return True, d[k]
    return False, None


ID_KEYS = ("point_ids", "pointIds", "ids", "id_list")
COORD_KEYS = (
    "voxel_coords",
    "voxelCoords",
    "voxel_coordinates",
    "coordinates",
    "coords",
    "voxels",
    "xyz",
)
FEATURE_KEYS = (
    "features",
    "feature_vectors",
    "featureVectors",
    "feats",
    "feature",
    "x",
)
ROW_CONTAINER_KEYS = ("points", "data", "items", "point_cloud", "pointcloud")

ROW_ID_KEYS = ("point_id", "pointId", "id", "point_number")
ROW_COORD_KEYS = (
    "voxel_coord",
    "voxelCoord",
    "voxel_coords",
    "voxel",
    "coord",
    "coords",
    "coordinate",
    "xyz",
    "position",
)
ROW_FEAT_KEYS = (
    "features",
    "feature",
    "feature_vector",
    "featureVector",
    "vector",
    "feat",
    "feats",
    "values",
    "embedding",
)


# --------------------------------------------------------------------------- #
# Point cloud upload
# --------------------------------------------------------------------------- #
def parse_pointcloud(payload: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """Return ``(point_ids, voxel_coords, features)`` raw Python lists."""
    if isinstance(payload, list):
        return _parse_rows(payload)

    if not isinstance(payload, dict):
        raise APIError(
            422,
            "INVALID_BODY",
            "request body must be a JSON object or a list of points",
        )

    has_ids, ids = _first_key(payload, ID_KEYS)
    has_coords, coords = _first_key(payload, COORD_KEYS)
    has_feats, feats = _first_key(payload, FEATURE_KEYS)

    if has_ids and has_coords and has_feats:
        if not isinstance(ids, list) or not isinstance(coords, list) or not isinstance(feats, list):
            raise APIError(422, "INVALID_BODY", "point_ids/voxel_coords/features must be arrays")
        return ids, coords, feats

    # Maybe {"points": [...rows...]} (or some other container key).
    for key in ROW_CONTAINER_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            return _parse_rows(rows)

    # Sometimes columns are nested under {"point_cloud": {...}}.
    for key in ("point_cloud", "pointcloud", "cloud"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            try:
                return parse_pointcloud(nested)
            except APIError:
                pass

    raise APIError(
        422,
        "INVALID_BODY",
        "could not find point_ids, voxel_coords and features "
        "(or a non-empty 'points' row list) in request body",
    )


def _parse_rows(rows: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    ids: list[Any] = []
    coords: list[Any] = []
    feats: list[Any] = []
    for i, row in enumerate(rows):
        if isinstance(row, dict):
            _, rid = _first_key(row, ROW_ID_KEYS)
            _, rc = _first_key(row, ROW_COORD_KEYS)
            _, rf = _first_key(row, ROW_FEAT_KEYS)
            if rid is None or rc is None or rf is None:
                raise APIError(
                    422,
                    "INVALID_POINT_ROW",
                    f"point row {i} must contain an id, a voxel coordinate and a feature vector",
                    {"row_keys": sorted(row.keys())},
                )
            ids.append(rid)
            coords.append(rc)
            feats.append(rf)
        elif isinstance(row, (list, tuple)) and len(row) == 3:
            rid, rc, rf = row
            ids.append(rid)
            coords.append(rc)
            feats.append(rf)
        else:
            raise APIError(
                422,
                "INVALID_POINT_ROW",
                f"point row {i} must be an object or [id, voxel_coord, features]",
            )
    if not ids:
        raise APIError(422, "EMPTY_POINT_CLOUD", "no points provided")
    return ids, coords, feats


# --------------------------------------------------------------------------- #
# Edit payloads
# --------------------------------------------------------------------------- #
EDIT_ID_KEYS = ("point_ids", "pointIds", "ids", "points", "target_ids")
MASK_KEYS = ("mask", "mask_ids", "masked", "hide", "hidden", "occlude")
RESTORE_KEYS = ("restore", "restore_ids", "restored", "show", "unmask", "reveal")
OPERATION_KEYS = ("operations", "ops", "updates", "edits", "actions")


def parse_id_list(value: Any, field_name: str = "point_ids") -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        # Allow [{"point_id": 1}, ...] shorthand.
        out: list[Any] = []
        for item in value:
            if isinstance(item, dict):
                _, rid = _first_key(item, ROW_ID_KEYS)
                if rid is None:
                    raise APIError(422, "INVALID_BODY", f"{field_name} entries need a point id")
                out.append(rid)
            else:
                out.append(item)
        return out
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return [value]
    raise APIError(422, "INVALID_BODY", f"{field_name} must be a list of point ids")


def parse_edit_payload(payload: Any) -> tuple[list[Any], list[Any], list[dict[str, Any]], Optional[str]]:
    """Return ``(mask_ids, restore_ids, operations, note)``."""
    if payload is None:
        return [], [], [], None
    if isinstance(payload, list):
        # Bare id list on /mask means mask them; caller decides endpoint flavor,
        # so signal via a special marker? Here treat bare list as mask only if
        # caller is mask endpoint. We return it as "ambiguous" through note?
        # Simpler: bare list -> mask list; restore endpoint wraps differently.
        return parse_id_list(payload), [], [], None
    if not isinstance(payload, dict):
        raise APIError(422, "INVALID_BODY", "edit body must be an object or a list of ids")

    note = payload.get("note")

    _, mask_val = _first_key(payload, MASK_KEYS)
    _, restore_val = _first_key(payload, RESTORE_KEYS)
    _, ops_val = _first_key(payload, OPERATION_KEYS)

    # {"point_ids": [...], "visible": false} convenience.
    _, ids_val = _first_key(payload, EDIT_ID_KEYS)
    visible = payload.get("visible")
    if mask_val is None and restore_val is None and ops_val is None and ids_val is not None:
        if visible is False or str(visible).lower() in {"0", "false", "masked"}:
            mask_val = ids_val
        elif visible is True or str(visible).lower() in {"1", "true", "visible"}:
            restore_val = ids_val
        else:
            # Endpoint-specific default handled by caller: return ids under both?
            # We stash via ops-less convention: return ids as mask candidate but
            # let caller override using parse_endpoint_edit.
            mask_val = ids_val

    mask_ids = parse_id_list(mask_val, "mask") if mask_val is not None else []
    restore_ids = parse_id_list(restore_val, "restore") if restore_val is not None else []

    operations: list[dict[str, Any]] = []
    if ops_val is not None:
        if not isinstance(ops_val, list):
            raise APIError(422, "INVALID_BODY", "operations must be a list")
        for op in ops_val:
            if not isinstance(op, dict):
                raise APIError(422, "INVALID_BODY", "each operation must be an object")
            operations.append(op)

    return mask_ids, restore_ids, operations, note


def parse_endpoint_edit(payload: Any, action: str) -> tuple[list[Any], list[dict[str, Any]], Optional[str]]:
    """Parse a /mask or /restore body.  Bare id lists are accepted."""
    if isinstance(payload, list):
        ids = parse_id_list(payload)
        return (ids, [], None) if action == "mask" else ([], ids, None)
    mask_ids, restore_ids, operations, note = parse_edit_payload(payload)
    if action == "mask":
        ids = mask_ids
        if not ids:
            _, raw = _first_key(payload if isinstance(payload, dict) else {}, EDIT_ID_KEYS)
            ids = parse_id_list(raw) if raw is not None else []
        return ids, operations, note
    else:
        ids = restore_ids
        if not ids:
            _, raw = _first_key(payload if isinstance(payload, dict) else {}, EDIT_ID_KEYS)
            ids = parse_id_list(raw) if raw is not None else []
        return ids, operations, note


# --------------------------------------------------------------------------- #
# Backward payload
# --------------------------------------------------------------------------- #
GRAD_KEYS = (
    "grad_output",
    "grad_outputs",
    "upstream_gradients",
    "upstream_grads",
    "output_grad",
    "output_gradients",
    "grad_values",
    "gradients",
    "grads",
    "grad",
    "gradient",
    "upstream_grad",
    "dy",
    "g",
)
VOXEL_INDEX_KEYS = ("voxel_indices", "voxel_index", "voxel_ids", "voxel_id", "indices")
VOXEL_COORD_COL_KEYS = ("voxel_coords", "voxel_coordinates", "coords", "coordinates")


def _as_grad_vector(raw: Any, c: int, where: str) -> np.ndarray:
    if not isinstance(raw, (list, tuple)):
        raise APIError(422, "INVALID_GRADIENT", f"{where} must be a list of {c} numbers")
    if len(raw) != c:
        raise APIError(
            422,
            "GRAD_SHAPE_MISMATCH",
            f"{where} has length {len(raw)}, expected channels={c}",
        )
    vals: list[float] = []
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise APIError(422, "INVALID_GRADIENT", f"{where} contains a non-numeric value {x!r}")
        fv = float(x)
        if not np.isfinite(fv):
            raise APIError(422, "INVALID_GRADIENT", f"{where} contains NaN/Infinity")
        vals.append(fv)
    return np.asarray(vals, dtype=np.float64)


def parse_coord_key(raw: Any, dim: int) -> Optional[tuple[int, ...]]:
    """Parse a voxel coordinate from list/tuple or a key like '1,2,3'."""
    if isinstance(raw, (list, tuple)):
        try:
            vals = tuple(int(x) for x in raw)
        except (TypeError, ValueError):
            return None
        return vals if len(vals) == dim else None
    if isinstance(raw, str):
        text = raw.strip().strip("[]()")
        parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
        if len(parts) != dim:
            return None
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            return None
    return None


def nonempty_voxel_indices(view: View) -> list[int]:
    states = view.resolved_voxel_states()
    out = [vidx for vidx in sorted(states.keys()) if states[vidx][1][0] >= 0]
    return out


def parse_backward_payload(
    payload: Any, view: View
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Return ``(grad_by_vidx, parse_meta)``.

    Malformed shapes raise 422.  Unknown / currently-empty voxel references
    are ignored by default (and reported in ``parse_meta``); with
    ``"strict": true`` they cause a 422 instead.
    """
    frame: Frame = view.frame
    c = frame.channels
    dim = frame.coord_dim
    strict = bool(payload.get("strict", False)) if isinstance(payload, dict) else False

    # raw_entries: list of (coord_or_None, index_or_None, vector, label)
    raw_entries: list[dict[str, Any]] = []
    positional_matrix: Optional[list[Any]] = None

    # ---- locate the gradient data ---------------------------------------- #
    if isinstance(payload, list):
        if payload and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in payload):
            # Bare 1-D vector.
            if len(payload) != c:
                raise APIError(422, "GRAD_SHAPE_MISMATCH", f"gradient length {len(payload)} != channels {c}")
            ne = nonempty_voxel_indices(view)
            if len(ne) != 1:
                raise APIError(
                    422,
                    "AMBIGUOUS_GRADIENT",
                    "bare gradient vector is only allowed when the view has exactly one non-empty voxel",
                )
            raw_entries.append({"vidx": ne[0], "vec": payload, "ref": frame.voxel_coord(ne[0])})
        else:
            _collect_item_list(payload, raw_entries)
    elif isinstance(payload, dict):
        # Columnar coords/indices + matrix?
        _, coord_col = _first_key(payload, VOXEL_COORD_COL_KEYS)
        _, index_col = _first_key(payload, VOXEL_INDEX_KEYS)
        _, grad_data = _first_key(payload, GRAD_KEYS)

        if grad_data is None:
            # Mapping form: {"gradients_by_voxel": {"0,0,0": [...]}}
            mapping = None
            for mk in ("gradients_by_voxel", "grads_by_voxel", "gradient_by_voxel", "by_voxel"):
                if isinstance(payload.get(mk), dict):
                    mapping = payload[mk]
                    break
            if mapping is not None:
                for key, vec in mapping.items():
                    raw_entries.append({"coord_ref": key, "vec": vec, "ref": key})
            else:
                # Maybe the whole body is coord-key -> vector.
                coord_like = {
                    k: v
                    for k, v in payload.items()
                    if parse_coord_key(k, dim) is not None and isinstance(v, (list, tuple))
                }
                reserved = set(
                    GRAD_KEYS
                    + VOXEL_COORD_COL_KEYS
                    + VOXEL_INDEX_KEYS
                    + ("view_id", "strict", "frame_id", "dense", "include_dense")
                )
                if coord_like and all(k not in reserved for k in coord_like):
                    for key, vec in coord_like.items():
                        raw_entries.append({"coord_ref": key, "vec": vec, "ref": key})
                else:
                    raise APIError(
                        422,
                        "INVALID_BODY",
                        "backward request must contain grad_output/grad_values "
                        "(list of {voxel_coord, grad}) or voxel-aligned columns",
                    )
        elif coord_col is not None or index_col is not None:
            if not isinstance(grad_data, (list, tuple)):
                raise APIError(422, "INVALID_GRADIENT", "gradient data must be an array")
            refs = coord_col if coord_col is not None else index_col
            using_coord = coord_col is not None

            single_coord = (
                using_coord
                and isinstance(refs, (list, tuple))
                and len(refs) == dim
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in refs)
            )
            single_index = (
                (not using_coord)
                and (
                    isinstance(refs, (int, str))
                    and not isinstance(refs, bool)
                )
            )
            if single_coord or single_index:
                # {"voxel_coord": [0,0,0], "grad": [..]} style.
                raw_entries.append(
                    {
                        "coord_ref" if using_coord else "index_ref": refs,
                        "vec": list(grad_data),
                        "ref": refs,
                    }
                )
            else:
                if not isinstance(refs, list) or len(refs) != len(grad_data):
                    raise APIError(
                        422,
                        "INVALID_BODY",
                        "voxel reference column must match gradient rows",
                    )
                for ref, vec in zip(refs, grad_data):
                    if using_coord:
                        raw_entries.append({"coord_ref": ref, "vec": vec, "ref": ref})
                    else:
                        raw_entries.append({"index_ref": ref, "vec": vec, "ref": ref})
        elif isinstance(grad_data, list):
            # Could be item list or a bare matrix.
            if grad_data and all(isinstance(x, dict) for x in grad_data):
                _collect_item_list(grad_data, raw_entries)
            elif grad_data and all(
                isinstance(x, (list, tuple))
                and x
                and all(isinstance(y, (int, float)) and not isinstance(y, bool) for y in x)
                for x in grad_data[:1]
            ):
                positional_matrix = grad_data
            elif not grad_data:
                positional_matrix = []
            else:
                # Mixed / dict rows? try item collection.
                _collect_item_list(grad_data, raw_entries)
        else:
            raise APIError(422, "INVALID_GRADIENT", "gradient data must be an array")
    else:
        raise APIError(422, "INVALID_BODY", "backward body must be a JSON object or list")

    # ---- positional matrix alignment ------------------------------------- #
    ignored_empty: list[Any] = []
    ignored_unknown: list[Any] = []
    applied: dict[int, np.ndarray] = {}

    if positional_matrix is not None:
        rows = positional_matrix
        ne = nonempty_voxel_indices(view)
        if len(rows) == len(ne):
            for vidx, vec in zip(ne, rows):
                applied[vidx] = _as_grad_vector(vec, c, f"gradient for voxel {frame.voxel_coord(vidx)}")
        elif len(rows) == frame.num_voxels:
            # Dense layout over the frame's original voxel order; empty ones ignored.
            for vidx, vec in enumerate(rows):
                w = view.get_voxel_state(vidx)[1]
                if w[0] < 0:
                    ignored_empty.append(frame.voxel_coord(vidx))
                    continue
                applied[vidx] = _as_grad_vector(vec, c, f"gradient for voxel {vidx}")
        else:
            raise APIError(
                422,
                "GRAD_ROW_COUNT_MISMATCH",
                f"got {len(rows)} gradient rows; expected {len(ne)} non-empty voxels "
                f"or {frame.num_voxels} frame voxels",
                {"nonempty_voxels": len(ne), "frame_voxels": frame.num_voxels},
            )

    # ---- resolve explicit references ------------------------------------- #
    seen: set[int] = set(applied.keys())
    duplicate: list[Any] = []
    for entry in raw_entries:
        vidx: Optional[int] = entry.get("vidx")
        ref = entry.get("ref")
        if vidx is None and "index_ref" in entry:
            try:
                cand = int(entry["index_ref"])
            except (TypeError, ValueError):
                raise APIError(422, "INVALID_VOXEL_REF", f"invalid voxel_index {entry['index_ref']!r}")
            if 0 <= cand < frame.num_voxels:
                vidx = cand
        if vidx is None and "coord_ref" in entry:
            coord_ref = entry["coord_ref"]
            if isinstance(coord_ref, int) and not isinstance(coord_ref, bool):
                if 0 <= coord_ref < frame.num_voxels:
                    vidx = coord_ref
            else:
                coord = parse_coord_key(coord_ref, dim)
                if coord is not None:
                    vidx = frame.coord_to_vidx.get(coord)
        if vidx is None:
            if strict:
                raise APIError(404, "VOXEL_NOT_FOUND", f"voxel {ref!r} does not exist in this frame")
            ignored_unknown.append(ref)
            continue

        winners = view.get_voxel_state(vidx)[1]
        if winners[0] < 0:
            if strict:
                raise APIError(422, "VOXEL_EMPTY", f"voxel {frame.voxel_coord(vidx)} is empty in view {view.view_id}")
            ignored_empty.append(frame.voxel_coord(vidx))
            continue

        if vidx in seen:
            duplicate.append(ref)
            continue
        seen.add(vidx)
        applied[vidx] = _as_grad_vector(
            entry["vec"], c, f"gradient for voxel {frame.voxel_coord(vidx)}"
        )

    if duplicate:
        raise APIError(
            422,
            "DUPLICATE_VOXEL_GRADIENT",
            "each non-empty voxel must appear at most once",
            {"duplicates": duplicate},
        )

    meta = {
        "applied_voxels": len(applied),
        "ignored_empty_voxels": ignored_empty,
        "ignored_unknown_voxels": ignored_unknown,
        "strict": strict,
    }
    return applied, meta


def _collect_item_list(items: list[Any], out: list[dict[str, Any]]) -> None:
    for i, item in enumerate(items):
        if isinstance(item, dict):
            _, coord = _first_key(item, ROW_COORD_KEYS + VOXEL_COORD_COL_KEYS)
            _, vidx = _first_key(item, VOXEL_INDEX_KEYS + ("id",))
            _, vec = _first_key(item, GRAD_KEYS + ROW_FEAT_KEYS)
            if vec is None:
                raise APIError(
                    422,
                    "INVALID_GRADIENT",
                    f"gradient item {i} needs a grad/gradient/values vector",
                )
            entry: dict[str, Any] = {"vec": vec, "ref": coord if coord is not None else vidx}
            if coord is not None:
                entry["coord_ref"] = coord
            if vidx is not None:
                entry["index_ref"] = vidx
            if coord is None and vidx is None:
                raise APIError(
                    422,
                    "INVALID_VOXEL_REF",
                    f"gradient item {i} needs voxel_coord or voxel_index",
                )
            out.append(entry)
        elif isinstance(item, (list, tuple)):
            # [coord, grad] or [vidx, grad]
            if len(item) == 2:
                ref, vec = item
                entry = {"vec": vec, "ref": ref}
                if isinstance(ref, int):
                    entry["index_ref"] = ref
                else:
                    entry["coord_ref"] = ref
                out.append(entry)
            else:
                raise APIError(422, "INVALID_GRADIENT_ITEM", f"gradient item {i} must be [voxel, grad]")
        else:
            raise APIError(422, "INVALID_GRADIENT_ITEM", f"gradient item {i} is invalid")


# --------------------------------------------------------------------------- #
# Optional raw body helper
# --------------------------------------------------------------------------- #
async def read_json_body(request: Any) -> Any:
    try:
        return await request.json()
    except Exception:
        # Starlette usually handles this, but be explicit for empty bodies.
        body = await request.body()
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise APIError(422, "INVALID_JSON", f"request body is not valid JSON: {exc}") from exc
