"""Voxel max-pooling engine with copy-on-write / delta views.

Core ideas
----------
* A :class:`Frame` is one immutable point cloud: unique point ids, integer
  voxel coordinates and equal-length feature rows.  Points are grouped into
  voxels; inside every voxel point indices are ordered by point id so that
  ``argmax`` naturally implements the "ties -> smallest point id" rule.
* A :class:`View` is an immutable visibility snapshot derived from another
  view.  A child view stores **only**:
    - visibility overrides for points whose visibility actually flipped
      (``K`` entries),
    - recomputed max values / winners for the dirty voxels containing those
      points (``D`` voxels * ``C`` channels).
  Clean voxel results and unchanged visibility are resolved by walking the
  parent chain (nearest ancestor wins).  Consequently old views stay
  valid forever and backward can use the winners that won "at that time".
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Optional

import numpy as np

# A voxel is considered empty iff its first channel winner is -1
# (every channel either has a winner or the voxel has no visible point).
NO_WINNER = -1

_INTEGER_RE = re.compile(r"^[+-]?\d+$")


class APIError(Exception):
    """Domain error that the HTTP layer maps to a 4xx response."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


# --------------------------------------------------------------------------- #
# Id helpers
# --------------------------------------------------------------------------- #
def canonical_point_id(raw: Any) -> Hashable:
    """Normalize a point id to an int or a str.

    Booleans are rejected (they are ints in Python but make terrible ids).
    """
    if isinstance(raw, bool):
        raise APIError(422, "INVALID_POINT_ID", "boolean is not a valid point id")
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise APIError(422, "INVALID_POINT_ID", "empty string is not a point id")
        return raw  # keep original (non-stripped) as canonical? use s? choose s
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    raise APIError(
        422,
        "INVALID_POINT_ID",
        f"point id must be an integer or string, got {type(raw).__name__}",
        {"value": raw},
    )


def equivalent_keys(cid: Hashable) -> list[Hashable]:
    """All lookup keys that should resolve to this canonical id.

    Numeric ids accept their string form and numeric strings accept the
    integer form, so ``1`` and ``"1"`` are interchangeable on lookup while
    still being rejected as duplicate ids inside one cloud.
    """
    keys = [cid]
    if isinstance(cid, int) and not isinstance(cid, bool):
        keys.append(str(cid))
    elif isinstance(cid, str) and _INTEGER_RE.match(cid.strip()):
        try:
            keys.append(int(cid.strip()))
        except ValueError:
            pass
    return keys


def lookup_key(raw: Any) -> Hashable:
    """Best-effort normalization of a request-supplied id for dict lookup."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if _INTEGER_RE.match(s):
            try:
                return int(s)
            except ValueError:
                return raw
        return raw
    return raw


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    frame_id: str
    point_ids: list[Hashable]                 # canonical id per upload index
    id_to_index: dict[Hashable, int]
    coords: np.ndarray                        # [N, D] int64
    features: np.ndarray                      # [N, C] float64
    voxel_keys: np.ndarray                    # [V, D] int64 lexicographic order
    coord_to_vidx: dict[tuple[int, ...], int]
    point_vidx: np.ndarray                     # [N]
    members_flat: np.ndarray                  # [N] pidx grouped by voxel, id asc
    offsets: np.ndarray                       # [V+1]
    channels: int
    coord_dim: int
    created_at: float = field(default_factory=time.time)

    # -- basic helpers ------------------------------------------------------ #
    @property
    def num_points(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_voxels(self) -> int:
        return int(self.voxel_keys.shape[0])

    def voxel_coord(self, vidx: int) -> list[int]:
        return [int(x) for x in self.voxel_keys[vidx]]

    def lookup_point(self, raw: Any) -> Optional[int]:
        """Return point index or None when the id is unknown."""
        key = lookup_key(raw)
        pidx = self.id_to_index.get(key)
        if pidx is not None:
            return pidx
        # direct fallback (e.g. non-numeric string with surrounding spacing)
        if isinstance(raw, str):
            return self.id_to_index.get(raw)
        return None

    def members(self, vidx: int) -> np.ndarray:
        return self.members_flat[self.offsets[vidx]: self.offsets[vidx + 1]]


def _fast_coordinate_array(raw: list[Any], n: int) -> Optional[np.ndarray]:
    """Vectorized integer-coordinate conversion, or None to use strict loop."""
    try:
        arr = np.asarray(raw)
    except (TypeError, ValueError):
        return None
    if arr.dtype == object or arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] == 0:
        return None
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            return None
        if np.all(arr == np.floor(arr)):
            return arr.astype(np.int64)
        return None
    return None  # bool / other dtypes -> strict fallback


def _fast_feature_array(raw: list[Any], n: int) -> Optional[np.ndarray]:
    """Vectorized float feature conversion with finite check."""
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] == 0:
        return None
    if not np.all(np.isfinite(arr)):
        raise APIError(
            422,
            "NON_FINITE_FEATURE",
            "features contain NaN/Infinity; finite values are required",
        )
    return arr


def build_frame(
    frame_id: str,
    raw_ids: list[Any],
    raw_coords: list[Any],
    raw_features: list[Any],
) -> Frame:
    """Validate normalized columns and construct a :class:`Frame`."""
    if not isinstance(raw_ids, list) or not raw_ids:
        raise APIError(422, "EMPTY_POINT_CLOUD", "point_ids must be a non-empty list")
    n = len(raw_ids)

    if len(raw_coords) != n:
        raise APIError(
            422,
            "LENGTH_MISMATCH",
            f"point_ids has {n} entries but voxel_coords has {len(raw_coords)}",
        )
    if len(raw_features) != n:
        raise APIError(
            422,
            "LENGTH_MISMATCH",
            f"point_ids has {n} entries but features has {len(raw_features)}",
        )

    # ---- ids --------------------------------------------------------------
    ids: list[Hashable] = []
    id_to_index: dict[Hashable, int] = {}
    for i, raw in enumerate(raw_ids):
        cid = canonical_point_id(raw)
        keys = equivalent_keys(cid)
        for key in keys:
            if key in id_to_index:
                raise APIError(
                    422,
                    "DUPLICATE_POINT_ID",
                    f"point id {raw!r} is duplicated (rows {id_to_index[key]} and {i})",
                    {"point_id": raw},
                )
        for key in keys:
            id_to_index[key] = i
        ids.append(cid)

    id_kinds = {("int" if isinstance(x, int) else "str") for x in ids}
    if len(id_kinds) > 1:
        raise APIError(
            422,
            "MIXED_POINT_ID_TYPES",
            "all point ids must have the same type (all integers or all strings)",
        )
    try:
        sorted(range(n), key=lambda i: ids[i])
    except TypeError as exc:  # pragma: no cover - defensive
        raise APIError(422, "UNORDERABLE_POINT_IDS", str(exc)) from exc

    # ---- coordinates ------------------------------------------------------
    # Vectorized fast path for the common homogeneous case; the element-wise
    # strict loop below remains as a fallback that emits precise 422 errors.
    coords = _fast_coordinate_array(raw_coords, n)
    dim: Optional[int] = int(coords.shape[1]) if coords is not None else None
    if coords is None:
        coords_rows: list[list[int]] = []
        for i, row in enumerate(raw_coords):
            if not isinstance(row, (list, tuple)):
                raise APIError(422, "INVALID_COORD", f"voxel_coords[{i}] must be a list")
            parsed: list[int] = []
            for x in row:
                if isinstance(x, bool):
                    raise APIError(422, "INVALID_COORD", f"voxel_coords[{i}] contains boolean")
                if isinstance(x, int):
                    parsed.append(int(x))
                elif isinstance(x, float) and x.is_integer():
                    parsed.append(int(x))
                else:
                    raise APIError(
                        422,
                        "NON_INTEGER_COORD",
                        f"voxel_coords[{i}]={list(row)!r} contains non-integer coordinate {x!r}",
                    )
            if dim is None:
                if len(parsed) == 0:
                    raise APIError(422, "INVALID_COORD", "voxel coordinate must be non-empty")
                dim = len(parsed)
            elif len(parsed) != dim:
                raise APIError(
                    422,
                    "COORD_DIM_MISMATCH",
                    f"voxel_coords[{i}] has dimension {len(parsed)}, expected {dim}",
                )
            coords_rows.append(parsed)
        coords = np.asarray(coords_rows, dtype=np.int64)

    # ---- features ---------------------------------------------------------
    if n == 0:  # pragma: no cover - guarded above
        raise APIError(422, "EMPTY_POINT_CLOUD", "at least one point is required")
    features = _fast_feature_array(raw_features, n)
    c_count: Optional[int] = int(features.shape[1]) if features is not None else None
    if features is None:
        feat_rows: list[list[float]] = []
        for i, row in enumerate(raw_features):
            if not isinstance(row, (list, tuple)):
                raise APIError(422, "INVALID_FEATURES", f"features[{i}] must be a list")
            vals: list[float] = []
            for x in row:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    raise APIError(
                        422,
                        "INVALID_FEATURES",
                        f"features[{i}] must contain only numbers, got {x!r}",
                    )
                fv = float(x)
                if not np.isfinite(fv):
                    raise APIError(
                        422,
                        "NON_FINITE_FEATURE",
                        f"features[{i}] contains NaN/Infinity; finite values are required",
                    )
                vals.append(fv)
            if c_count is None:
                if len(vals) == 0:
                    raise APIError(422, "EMPTY_FEATURE_VECTOR", "feature vector must be non-empty")
                c_count = len(vals)
            elif len(vals) != c_count:
                raise APIError(
                    422,
                    "FEATURE_DIM_MISMATCH",
                    f"features[{i}] has length {len(vals)}, expected {c_count}",
                )
            feat_rows.append(vals)
        features = np.asarray(feat_rows, dtype=np.float64)

    channels = int(c_count)  # type: ignore[arg-type]
    coord_dim = int(dim)  # type: ignore[arg-type]

    # ---- voxel grouping ---------------------------------------------------
    voxel_keys, inverse, counts = _unique_voxels(coords)
    num_voxels = int(voxel_keys.shape[0])

    # Sort points by (voxel index, point id).  Within a voxel the ascending
    # point-id order is what makes argmax tie-breaking pick the smallest id.
    order = sorted(range(n), key=lambda i: (int(inverse[i]), ids[i]))
    members_flat = np.asarray(order, dtype=np.int64)

    counts64 = np.asarray(counts, dtype=np.int64).reshape(-1)
    offsets = np.empty(num_voxels + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts64, out=offsets[1:])

    coord_to_vidx: dict[tuple[int, ...], int] = {}
    for v in range(num_voxels):
        coord_to_vidx[tuple(int(x) for x in voxel_keys[v])] = v

    return Frame(
        frame_id=frame_id,
        point_ids=ids,
        id_to_index=id_to_index,
        coords=coords,
        features=features,
        voxel_keys=voxel_keys,
        coord_to_vidx=coord_to_vidx,
        point_vidx=inverse.astype(np.int64, copy=False),
        members_flat=members_flat,
        offsets=offsets,
        channels=channels,
        coord_dim=coord_dim,
    )


def _unique_voxels(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """np.unique over rows with a fallback for unusual numpy builds."""
    keys, inverse, counts = np.unique(
        coords, axis=0, return_inverse=True, return_counts=True
    )
    return keys.astype(np.int64, copy=False), inverse.reshape(-1), counts


# --------------------------------------------------------------------------- #
# Grouped max pooling
# --------------------------------------------------------------------------- #
def grouped_max_pool(
    features: np.ndarray,
    group_ids: np.ndarray,
    num_groups: int,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per group / channel maximum and first-winning row.

    Rows must be ordered by (group, ascending point id within group).

    Returns
    -------
    values : [G, C] float64  (-inf for groups with no visible row)
    winner_row : [G, C] int64, ``M`` sentinel when there is no winner.
    """
    m, c = features.shape
    values = np.full((num_groups, c), -np.inf, dtype=np.float64)
    masked = np.where(visible[:, None], features, -np.inf)
    if m:
        np.maximum.at(values, group_ids, masked)

        is_winner = visible[:, None] & (features == values[group_ids])
        row_index = np.arange(m, dtype=np.int64)[:, None]
        candidates = np.where(is_winner, row_index, m)
        winner_row = np.full((num_groups, c), m, dtype=np.int64)
        np.minimum.at(winner_row, group_ids, candidates)
    else:
        winner_row = np.full((num_groups, c), 0, dtype=np.int64)
    return values, winner_row


# --------------------------------------------------------------------------- #
# View (immutable delta node)
# --------------------------------------------------------------------------- #
@dataclass
class View:
    view_id: int
    frame: Frame
    parent: Optional["View"]
    depth: int
    vis_delta: dict[int, bool]                 # pidx -> effective visibility
    dirty_vidx: np.ndarray                     # [D]
    values: np.ndarray                         # [D, C]
    winner_pidx: np.ndarray                    # [D, C], NO_WINNER = -1
    dirty_pos: dict[int, int]
    changed_pidx: list[int]
    note: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    # NOTE: a View deliberately does NOT retain request payloads (requested
    # id lists can be huge even when nothing actually changes).  Its resident
    # delta consists solely of ``vis_delta`` (K entries), ``changed_pidx``
    # (K) and the dirty voxel arrays (D*C slots).

    # -- visibility resolution --------------------------------------------- #
    def resolve_visibility(self, wanted: set[int]) -> dict[int, bool]:
        """Effective visibility for the given point indices in this view.

        Walks the delta chain nearest-first; points never mentioned default
        to visible (the initial state).
        """
        remaining = set(wanted)
        result: dict[int, bool] = {}
        cur: Optional[View] = self
        while cur is not None and remaining:
            delta = cur.vis_delta
            if delta:
                # Iterate the smaller side to keep this cheap.
                if len(delta) <= len(remaining):
                    for pidx, state in delta.items():
                        if pidx in remaining:
                            result[pidx] = state
                            remaining.remove(pidx)
                else:
                    for pidx in list(remaining):
                        state = delta.get(pidx)
                        if state is not None or pidx in delta:
                            result[pidx] = bool(state)
                            remaining.remove(pidx)
            cur = cur.parent
        for pidx in remaining:
            result[pidx] = True
        return result

    # -- voxel state resolution -------------------------------------------- #
    def get_voxel_state(
        self, vidx: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (values[C], winner_pidx[C], source_view_id) for a voxel."""
        cur: Optional[View] = self
        while cur is not None:
            pos = cur.dirty_pos.get(vidx)
            if pos is not None:
                return cur.values[pos], cur.winner_pidx[pos], cur.view_id
            cur = cur.parent
        # Impossible for a frame voxel: the root view stores all voxels.
        raise KeyError(f"voxel {vidx} missing from view chain {self.view_id}")

    def resolved_voxel_states(
        self,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
        """Nearest-ancestor resolution for every voxel (used for full output)."""
        states: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        cur: Optional[View] = self
        while cur is not None:
            for pos, vidx_raw in enumerate(cur.dirty_vidx):
                vidx = int(vidx_raw)
                if vidx not in states:
                    states[vidx] = (
                        cur.values[pos],
                        cur.winner_pidx[pos],
                        cur.view_id,
                    )
            cur = cur.parent
        return states

    # -- small stats -------------------------------------------------------- #
    @property
    def changed_points(self) -> int:
        return len(self.changed_pidx)

    @property
    def recomputed_voxels(self) -> int:
        return int(self.dirty_vidx.shape[0])

    @property
    def recomputed_channels(self) -> int:
        return int(self.dirty_vidx.shape[0]) * self.frame.channels

    def delta_stats(self) -> dict[str, int]:
        d = int(self.dirty_vidx.shape[0])
        c = self.frame.channels
        return {
            "visibility_entries": len(self.vis_delta),
            "changed_points": len(self.changed_pidx),
            "voxel_result_entries": d,
            "channel_slots_per_array": d * c,
            "stored_max_values": d * c,
            "stored_winner_ids": d * c,
            "numpy_bytes_values": int(self.values.nbytes),
            "numpy_bytes_winners": int(self.winner_pidx.nbytes),
            "numpy_bytes_dirty_index": int(self.dirty_vidx.nbytes),
            "total_delta_numpy_bytes": int(
                self.values.nbytes
                + self.winner_pidx.nbytes
                + self.dirty_vidx.nbytes
            ),
        }


# --------------------------------------------------------------------------- #
# Store / service
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.frames: dict[str, Frame] = {}
        self.views: dict[int, View] = {}
        self.frame_order: list[str] = []
        self._frame_seq = 0
        self._view_seq = 0
        self.cumulative_voxel_recomputations = 0
        self.edit_count = 0

    # -- frame / root view -------------------------------------------------- #
    def create_frame(
        self, raw_ids: list[Any], raw_coords: list[Any], raw_features: list[Any]
    ) -> tuple[Frame, View]:
        with self.lock:
            self._frame_seq += 1
            frame_id = f"f{self._frame_seq}"
            frame = build_frame(frame_id, raw_ids, raw_coords, raw_features)
            self.frames[frame_id] = frame
            self.frame_order.append(frame_id)

            root = self._build_root_view(frame)
            self.views[root.view_id] = root
            return frame, root

    def _alloc_view_id(self) -> int:
        self._view_seq += 1
        return self._view_seq

    def _build_root_view(self, frame: Frame) -> View:
        vid = self._alloc_view_id()
        v = frame.num_voxels
        group_ids = np.repeat(np.arange(v, dtype=np.int64), np.diff(frame.offsets))
        visible = np.ones(frame.num_points, dtype=bool)
        values, winner_rows = grouped_max_pool(
            frame.features[frame.members_flat],
            group_ids,
            v,
            visible,
        )
        winner_pidx = np.full((v, frame.channels), NO_WINNER, dtype=np.int64)
        has = winner_rows < frame.num_points
        winner_pidx[has] = frame.members_flat[winner_rows[has]]

        dirty_vidx = np.arange(v, dtype=np.int64)
        view = View(
            view_id=vid,
            frame=frame,
            parent=None,
            depth=0,
            vis_delta={},
            dirty_vidx=dirty_vidx,
            values=values,
            winner_pidx=winner_pidx,
            dirty_pos={i: i for i in range(v)},
            changed_pidx=[],
            note="initial view: all points visible",
        )
        self.cumulative_voxel_recomputations += v
        return view

    # -- lookup ------------------------------------------------------------- #
    def get_frame(self, frame_id: str) -> Frame:
        frame = self.frames.get(frame_id)
        if frame is None:
            raise APIError(404, "FRAME_NOT_FOUND", f"frame {frame_id!r} does not exist")
        return frame

    def resolve_view(self, ref: Any, frame_id: Optional[str] = None) -> View:
        """Resolve a path/body view reference.

        Accepts integer ids and forms like ``"3"``, ``"v3"``,
        ``"view-3"``, ``"view_003"``; frame-scoped callers may also pass
        ``"initial"`` / ``"root"`` / ``"latest"``.
        """
        with self.lock:
            if isinstance(ref, int) and not isinstance(ref, bool):
                view = self.views.get(ref)
                if view is not None:
                    return view
                raise APIError(404, "VIEW_NOT_FOUND", f"view {ref} does not exist")

            if not isinstance(ref, str):
                # maybe numeric passed as e.g. 3.0
                if isinstance(ref, float) and ref.is_integer():
                    return self.resolve_view(int(ref), frame_id)
                raise APIError(404, "VIEW_NOT_FOUND", f"view {ref!r} does not exist")

            text = ref.strip()
            low = text.lower()

            if low in {"initial", "root", "all-visible", "default"}:
                if frame_id is None:
                    raise APIError(
                        400,
                        "FRAME_REQUIRED",
                        f"view reference {text!r} is frame-scoped; provide a frame id",
                    )
                frame = self.get_frame(frame_id)
                root = self._root_view_of(frame)
                return root

            if low in {"latest", "current", "head"}:
                if frame_id is None:
                    raise APIError(
                        400,
                        "FRAME_REQUIRED",
                        f"view reference {text!r} is frame-scoped; provide a frame id",
                    )
                frame = self.get_frame(frame_id)
                return self._latest_view_of(frame)

            m = re.match(r"^(?:view[-_]?|v)?0*(\d+)$", low)
            if m:
                vid = int(m.group(1))
                view = self.views.get(vid)
                if view is not None:
                    if frame_id is not None and view.frame.frame_id != frame_id:
                        raise APIError(
                            404,
                            "VIEW_NOT_IN_FRAME",
                            f"view {vid} does not belong to frame {frame_id}",
                        )
                    return view

            raise APIError(404, "VIEW_NOT_FOUND", f"view {ref!r} does not exist")

    def _root_view_of(self, frame: Frame) -> View:
        for view in self.views.values():
            if view.frame is frame and view.parent is None:
                return view
        raise APIError(500, "ROOT_VIEW_MISSING", "frame has no root view")

    def _latest_view_of(self, frame: Frame) -> View:
        latest: Optional[View] = None
        for view in self.views.values():
            if view.frame is frame and (latest is None or view.view_id > latest.view_id):
                latest = view
        if latest is None:
            raise APIError(404, "VIEW_NOT_FOUND", "frame has no views")
        return latest

    # -- edit --------------------------------------------------------------- #
    def apply_edit(
        self,
        base: View,
        mask_raw: Optional[list[Any]] = None,
        restore_raw: Optional[list[Any]] = None,
        operations: Optional[list[dict[str, Any]]] = None,
        note: Optional[str] = None,
    ) -> View:
        """Create a child view with batched mask/restore operations.

        The whole request is rejected (without creating any view) when it
        references an unknown point id.
        """
        mask_raw = list(mask_raw or [])
        restore_raw = list(restore_raw or [])
        operations = list(operations or [])
        frame = base.frame

        # ---- resolve ids, collecting every unknown id first -------------- #
        unknown: list[Any] = []

        def to_pidx(raw_id: Any) -> Optional[int]:
            pidx = frame.lookup_point(raw_id)
            if pidx is None:
                unknown.append(raw_id)
            return pidx

        mask_pidx = [p for p in (to_pidx(x) for x in mask_raw) if p is not None]
        restore_pidx = [p for p in (to_pidx(x) for x in restore_raw) if p is not None]

        op_desired: list[tuple[int, bool]] = []
        for op in operations:
            raw_id = op.get("point_id", op.get("id", op.get("pointId")))
            action = str(
                op.get("action", op.get("op", op.get("operation", "")))
            ).lower()
            pidx = to_pidx(raw_id)
            if pidx is None:
                continue
            if action in {"mask", "hide", "occlude", "delete", "remove", "0", "false"}:
                op_desired.append((pidx, False))
            elif action in {"restore", "show", "unmask", "reveal", "add", "1", "true"}:
                op_desired.append((pidx, True))
            else:
                raise APIError(
                    422,
                    "UNKNOWN_OPERATION",
                    f"operation action {action!r} is not 'mask' or 'restore'",
                    {"operation": op},
                )

        if unknown:
            # Atomic rejection: no view is allocated or registered.
            raise APIError(
                400,
                "UNKNOWN_POINT_IDS",
                f"{len(unknown)} point id(s) are not part of frame {frame.frame_id}; "
                "the entire edit was rejected and no view was created",
                {"unknown_point_ids": unknown},
            )

        # ---- desired visibility vs. base ---------------------------------- #
        wanted = set(mask_pidx) | set(restore_pidx) | {p for p, _ in op_desired}
        base_vis = base.resolve_visibility(wanted)
        desired = dict(base_vis)

        # Combined object form: mask first, restore wins on overlap.
        for pidx in mask_pidx:
            desired[pidx] = False
        for pidx in restore_pidx:
            desired[pidx] = True
        # Explicit operation list applied last, in order.
        for pidx, state in op_desired:
            desired[pidx] = state

        changes: dict[int, bool] = {
            pidx: desired[pidx]
            for pidx in wanted
            if desired[pidx] != base_vis[pidx]
        }
        changed_pidx = sorted(changes.keys(), key=lambda p: frame.point_ids[p])

        dirty_vidx = np.unique(frame.point_vidx[np.asarray(changed_pidx, dtype=np.int64)]) \
            if changed_pidx else np.array([], dtype=np.int64)

        # ---- construct child delta ---------------------------------------- #
        with self.lock:
            view_id = self._alloc_view_id()
            child = View(
                view_id=view_id,
                frame=frame,
                parent=base,
                depth=base.depth + 1,
                vis_delta=changes,
                dirty_vidx=dirty_vidx.astype(np.int64),
                values=np.zeros((len(dirty_vidx), frame.channels), dtype=np.float64),
                winner_pidx=np.full(
                    (len(dirty_vidx), frame.channels), NO_WINNER, dtype=np.int64
                ),
                dirty_pos={int(v): i for i, v in enumerate(dirty_vidx)},
                changed_pidx=changed_pidx,
                note=note,
            )

            if len(dirty_vidx):
                lengths = frame.offsets[dirty_vidx + 1] - frame.offsets[dirty_vidx]
                pidx_rows = np.concatenate(
                    [frame.members(int(v)) for v in dirty_vidx]
                )
                group_ids = np.repeat(
                    np.arange(len(dirty_vidx), dtype=np.int64), lengths
                )
                vis_state = child.resolve_visibility(set(int(p) for p in pidx_rows))
                visible = np.asarray(
                    [vis_state[int(p)] for p in pidx_rows], dtype=bool
                )
                vals, winner_rows = grouped_max_pool(
                    frame.features[pidx_rows],
                    group_ids,
                    len(dirty_vidx),
                    visible,
                )
                child.values = vals
                has = winner_rows < len(pidx_rows)
                child.winner_pidx[has] = pidx_rows[winner_rows[has]]

            self.views[view_id] = child
            self.edit_count += 1
            self.cumulative_voxel_recomputations += len(dirty_vidx)
            return child

    # -- backward ----------------------------------------------------------- #
    def backward(
        self,
        view: View,
        grad_by_vidx: dict[int, np.ndarray],
    ) -> dict[str, Any]:
        """Route upstream voxel gradients to the winning point per channel."""
        frame = view.frame
        n, c = frame.num_points, frame.channels
        dense = np.zeros((n, c), dtype=np.float64)

        routed_pidx: list[int] = []
        routed_chan: list[int] = []
        routed_val: list[float] = []
        voxel_of_pidx = frame.point_vidx

        if grad_by_vidx:
            vidxs = sorted(grad_by_vidx.keys())
            winners = np.stack([view.get_voxel_state(v)[1] for v in vidxs])  # [K,C]
            grads = np.stack([grad_by_vidx[v] for v in vidxs]).astype(np.float64)
            k = len(vidxs)
            chan_grid = np.broadcast_to(np.arange(c, dtype=np.int64), (k, c))
            mask = winners >= 0
            p_sel = winners[mask]
            c_sel = chan_grid[mask]
            v_sel = np.broadcast_to(
                np.asarray(vidxs, dtype=np.int64)[:, None], (k, c)
            )[mask]
            g_sel = grads[mask]
            np.add.at(dense, (p_sel, c_sel), g_sel)

            routed_pidx = [int(x) for x in p_sel]
            routed_chan = [int(x) for x in c_sel]
            routed_val = [float(x) for x in g_sel]
            routed_vidx = [int(x) for x in v_sel]
        else:
            routed_vidx = []

        # Unique set of points that received at least one channel route.
        receiving = sorted(set(routed_pidx), key=lambda p: frame.point_ids[p])

        point_gradients: list[dict[str, Any]] = []
        gradients_by_point: dict[str, list[float]] = {}
        grad_point_ids: list[Any] = []
        grad_values: list[list[float]] = []
        for pidx in receiving:
            vec = [float(x) for x in dense[pidx]]
            cid = frame.point_ids[pidx]
            key = str(cid)
            gradients_by_point[key] = vec
            grad_point_ids.append(cid)
            grad_values.append(vec)
            point_gradients.append(
                {
                    "point_id": cid,
                    "point_index": pidx,
                    "voxel_index": int(voxel_of_pidx[pidx]),
                    "voxel_coord": frame.voxel_coord(int(voxel_of_pidx[pidx])),
                    "grad": vec,
                    "gradient": vec,
                }
            )

        sparse_entries = [
            {
                "point_id": frame.point_ids[p],
                "point_index": p,
                "channel": ch,
                "value": val,
                "voxel_index": vx,
            }
            for p, ch, val, vx in zip(
                routed_pidx, routed_chan, routed_val, routed_vidx
            )
        ]
        # Sort sparse entries deterministically by (point id order, channel).
        sparse_entries.sort(key=lambda e: (e["point_index"], e["channel"]))

        return {
            "dense_gradients": dense.tolist(),
            "point_ids": list(frame.point_ids),
            "point_gradients": point_gradients,
            "gradients_by_point": gradients_by_point,
            "grad_point_ids": grad_point_ids,
            "grad_values": grad_values,
            "point_grads": grad_values,
            "point_grad_values": grad_values,
            "sparse_gradients": sparse_entries,
            "sparse": {
                "point_ids": [e["point_id"] for e in sparse_entries],
                "channels": [e["channel"] for e in sparse_entries],
                "values": [e["value"] for e in sparse_entries],
            },
            "routed_channels": len(routed_pidx),
            "receiving_point_count": len(receiving),
        }


# A process-wide default store used by the HTTP application.
store = Store()
