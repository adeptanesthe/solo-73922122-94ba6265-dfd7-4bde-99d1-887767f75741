"""Voxel max-pooling service with lightweight occlusion views.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs .
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from .core import APIError, store
from .demo import run_walkthrough
from .parsing import (
    parse_backward_payload,
    parse_edit_payload,
    parse_endpoint_edit,
    parse_pointcloud,
)
from .responses import frame_detail, frame_summary, view_output

logger = logging.getLogger("voxel-pool-api")

app = FastAPI(
    title="Voxel MaxPool Occlusion-View API",
    version="1.0.0",
    description=(
        "Point-cloud voxel max pooling for segmentation training.\n\n"
        "* Upload a cloud with unique point ids, integer voxel coords and "
        "equal-length feature vectors (initial view: every point visible).\n"
        "* Mask / restore point batches on **any** view; each edit returns a "
        "**new view id** plus every non-empty voxel's channel-wise maxima.\n"
        "* Ties are resolved to the **smallest point id** per channel.\n"
        "* Only voxels containing points whose visibility actually changed are "
        "recomputed; the new view stores just that delta.\n"
        "* Backward submits upstream voxel gradients for a view and returns "
        "sparse point gradients routed to the point ids that won at that time.\n"
        "* Edits mentioning unknown point ids reject the entire request.\n\n"
        "Start with `POST /demo/walkthrough` or expand the examples below."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(APIError)
async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": str(exc)},
    )


# --------------------------------------------------------------------------- #
# OpenAPI examples
# --------------------------------------------------------------------------- #
UPLOAD_EXAMPLES = {
    "columnar": {
        "summary": "Columnar point cloud (5 points, 2 channels)",
        "value": {
            "point_ids": [1, 2, 3, 4, 5],
            "voxel_coords": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
            "features": [[10, 0], [10, 7], [3, 5], [2, 9], [8, 1]],
        },
    },
    "rows": {
        "summary": "Row-object style",
        "value": {
            "points": [
                {"point_id": 1, "voxel_coord": [0, 0, 0], "features": [10, 0]},
                {"point_id": 2, "voxel_coord": [0, 0, 0], "features": [10, 7]},
            ]
        },
    },
}

MASK_EXAMPLES = {
    "mask": {
        "summary": "Mask points 1 and 3",
        "value": {"point_ids": [1, 3]},
    },
    "mask_alias": {
        "summary": "Explicit mask list",
        "value": {"mask": [1, 3], "note": "random occlusion batch"},
    },
}

RESTORE_EXAMPLES = {
    "restore": {
        "summary": "Restore point 1",
        "value": {"point_ids": [1]},
    }
}

EDIT_EXAMPLES = {
    "combined": {
        "summary": "Mask some and restore others atomically",
        "value": {"mask": [2], "restore": [1], "note": "swap occlusion"},
    },
    "operations": {
        "summary": "Ordered operation list",
        "value": {
            "operations": [
                {"point_id": 1, "action": "mask"},
                {"point_id": 2, "action": "restore"},
            ]
        },
    },
}

BACKWARD_EXAMPLES = {
    "objects": {
        "summary": "Gradient per voxel coordinate",
        "value": {
            "grad_output": [
                {"voxel_coord": [0, 0, 0], "grad": [1.0, 2.0]},
                {"voxel_coord": [1, 0, 0], "grad": [3.0, 4.0]},
            ]
        },
    },
    "indices": {
        "summary": "Gradient per voxel index (from forward output)",
        "value": {
            "grad_output": [
                {"voxel_index": 0, "grad": [1.0, 2.0]},
                {"voxel_index": 1, "grad": [3.0, 4.0]},
            ]
        },
    },
    "columnar": {
        "summary": "Columnar form",
        "value": {
            "voxel_coords": [[0, 0, 0], [1, 0, 0]],
            "grad_values": [[1.0, 2.0], [3.0, 4.0]],
        },
    },
}


# --------------------------------------------------------------------------- #
# Meta routes
# --------------------------------------------------------------------------- #
@router.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "voxel-maxpool-occlusion-view-api",
        "frames": len(store.frames),
        "views": len(store.views),
        "edits": store.edit_count,
        "cumulative_voxel_recomputations": store.cumulative_voxel_recomputations,
    }


@router.get("/stats", tags=["meta"])
def stats() -> dict[str, Any]:
    return {
        "frames": len(store.frames),
        "views": len(store.views),
        "edits": store.edit_count,
        "cumulative_voxel_recomputations": store.cumulative_voxel_recomputations,
        "view_ids": sorted(store.views.keys()),
        "frame_ids": list(store.frames.keys()),
    }


# --------------------------------------------------------------------------- #
# Point clouds / frames
# --------------------------------------------------------------------------- #
@router.post("/pointclouds", tags=["point clouds"])
def create_pointcloud(
    payload: Any = Body(..., openapi_examples=UPLOAD_EXAMPLES),
) -> dict[str, Any]:
    raw_ids, raw_coords, raw_features = parse_pointcloud(payload)
    frame, root = store.create_frame(raw_ids, raw_coords, raw_features)
    summary = frame_summary(frame, root.view_id)
    output = view_output(root)
    body = dict(summary)
    body["initial_view"] = output
    body.update(
        {
            "view_id": root.view_id,
            "initial_view_id": root.view_id,
            "view": root.view_id,
            "new_view_id": root.view_id,
            "voxels": output["voxels"],
            "voxel_coords": output["voxel_coords"],
            "max_values": output["max_values"],
            "values": output["max_values"],
            "winner_point_ids": output["winner_point_ids"],
            "nonempty_voxel_count": output["nonempty_voxel_count"],
        }
    )
    return body


@router.get("/pointclouds", tags=["point clouds"])
def list_pointclouds() -> dict[str, Any]:
    return {
        "frames": [frame_summary(store.frames[fid]) for fid in store.frame_order],
        "count": len(store.frames),
    }


@router.get("/pointclouds/{frame_id}", tags=["point clouds"])
def get_pointcloud(frame_id: str) -> dict[str, Any]:
    return frame_detail(store.get_frame(frame_id))


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
@router.get("/views", tags=["views"])
def list_views(frame_id: str | None = None) -> dict[str, Any]:
    views = list(store.views.values())
    if frame_id is not None:
        frame = store.get_frame(frame_id)
        views = [v for v in views if v.frame is frame]
    views.sort(key=lambda v: v.view_id)
    return {
        "views": [
            {
                "view_id": v.view_id,
                "frame_id": v.frame.frame_id,
                "parent_view_id": v.parent.view_id if v.parent else None,
                "depth": v.depth,
                "is_initial": v.parent is None,
                "changed_points": v.changed_points,
                "recomputed_voxels": v.recomputed_voxels,
            }
            for v in views
        ],
        "count": len(views),
    }


@router.get("/views/{view_id}", tags=["views"])
def get_view(view_id: str) -> dict[str, Any]:
    return view_output(store.resolve_view(view_id))


@router.get("/views/{view_id}/output", tags=["views"])
def get_view_output(view_id: str) -> dict[str, Any]:
    return view_output(store.resolve_view(view_id))


def _edit_response(base_ref: str, payload: Any, action: str | None) -> dict[str, Any]:
    base = store.resolve_view(base_ref)
    overlap: list[Any] = []
    if action in {"mask", "restore"}:
        ids, operations, note = parse_endpoint_edit(payload, action)
        if action == "mask":
            child = store.apply_edit(base, mask_raw=ids, operations=operations, note=note)
        else:
            child = store.apply_edit(base, restore_raw=ids, operations=operations, note=note)
    else:
        mask_ids, restore_ids, operations, note = parse_edit_payload(payload)
        child = store.apply_edit(
            base,
            mask_raw=mask_ids,
            restore_raw=restore_ids,
            operations=operations,
            note=note,
        )
        frame = base.frame
        mset = {p for raw in mask_ids if (p := frame.lookup_point(raw)) is not None}
        rset = {p for raw in restore_ids if (p := frame.lookup_point(raw)) is not None}
        both = sorted(mset & rset, key=lambda p: frame.point_ids[p])
        overlap = [frame.point_ids[p] for p in both]
    out = view_output(child)
    out["new_view_id"] = child.view_id
    if action == "edit":
        out["overlap_point_ids"] = overlap
        if overlap:
            out["overlap_resolution"] = "restore wins"
    return out


def _view_ref_from_body(payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise APIError(422, "VIEW_REQUIRED", "a JSON object with view_id is required")
    ref = payload.get("view_id", payload.get("viewId", payload.get("id")))
    if ref is None:
        raise APIError(422, "VIEW_REQUIRED", "view_id is required in the request body")
    return ref


@router.post("/mask", tags=["views: edit"])
def mask_by_body(
    payload: dict[str, Any] = Body(
        ...,
        openapi_examples={
            "body": {
                "summary": "Mask points on a view referenced in the body",
                "value": {"view_id": 1, "point_ids": [1, 3]},
            }
        },
    ),
) -> dict[str, Any]:
    ref = _view_ref_from_body(payload)
    return _edit_response(str(ref), payload, "mask")


@router.post("/restore", tags=["views: edit"])
def restore_by_body(
    payload: dict[str, Any] = Body(
        ...,
        openapi_examples={
            "body": {
                "summary": "Restore points on a view referenced in the body",
                "value": {"view_id": 2, "point_ids": [1]},
            }
        },
    ),
) -> dict[str, Any]:
    ref = _view_ref_from_body(payload)
    return _edit_response(str(ref), payload, "restore")


@router.post("/edit", tags=["views: edit"])
def edit_by_body(
    payload: dict[str, Any] = Body(
        ...,
        openapi_examples={
            "body": {
                "summary": "Combined edit with view id in body",
                "value": {"view_id": 1, "mask": [2], "restore": [1]},
            }
        },
    ),
) -> dict[str, Any]:
    ref = _view_ref_from_body(payload)
    return _edit_response(str(ref), payload, "edit")


@router.post("/views/{view_id}/mask", tags=["views: edit"])
def mask_view(
    view_id: str,
    payload: Any = Body(default=None, openapi_examples=MASK_EXAMPLES),
) -> dict[str, Any]:
    return _edit_response(view_id, payload if payload is not None else {}, "mask")


@router.post("/views/{view_id}/restore", tags=["views: edit"])
def restore_view(
    view_id: str,
    payload: Any = Body(default=None, openapi_examples=RESTORE_EXAMPLES),
) -> dict[str, Any]:
    return _edit_response(view_id, payload if payload is not None else {}, "restore")


@router.post("/views/{view_id}/edit", tags=["views: edit"])
def edit_view(
    view_id: str,
    payload: Any = Body(default=None, openapi_examples=EDIT_EXAMPLES),
) -> dict[str, Any]:
    return _edit_response(view_id, payload if payload is not None else {}, "edit")


@router.post("/views/{view_id}/visibility", include_in_schema=False)
def visibility_view(view_id: str, payload: Any = Body(default={})) -> dict[str, Any]:
    return _edit_response(view_id, payload or {}, "edit")


# Frame-scoped variants, allowing "initial" / "latest" references.
@router.post("/pointclouds/{frame_id}/views/{view_ref}/mask", tags=["views: edit"])
def mask_frame_view(
    frame_id: str,
    view_ref: str,
    payload: Any = Body(default=None, openapi_examples=MASK_EXAMPLES),
) -> dict[str, Any]:
    store.get_frame(frame_id)
    base = store.resolve_view(view_ref, frame_id=frame_id)
    return _edit_response(str(base.view_id), payload if payload is not None else {}, "mask")


@router.post("/pointclouds/{frame_id}/views/{view_ref}/restore", tags=["views: edit"])
def restore_frame_view(
    frame_id: str,
    view_ref: str,
    payload: Any = Body(default=None, openapi_examples=RESTORE_EXAMPLES),
) -> dict[str, Any]:
    store.get_frame(frame_id)
    base = store.resolve_view(view_ref, frame_id=frame_id)
    return _edit_response(str(base.view_id), payload if payload is not None else {}, "restore")


@router.get("/pointclouds/{frame_id}/views/{view_ref}/output", tags=["views"])
def frame_view_output(frame_id: str, view_ref: str) -> dict[str, Any]:
    view = store.resolve_view(view_ref, frame_id=frame_id)
    return view_output(view)


# --------------------------------------------------------------------------- #
# Backward
# --------------------------------------------------------------------------- #
# Full N*C dense gradients are attached automatically only for small clouds;
# larger responses stay truly sparse unless the caller opts in.
DENSE_CELL_LIMIT = 65_536


def _wants_dense(payload: Any, query_value: bool | None) -> bool | None:
    if query_value is not None:
        return query_value
    if isinstance(payload, dict):
        if "dense" in payload:
            value = payload["dense"]
        elif "include_dense" in payload:
            value = payload["include_dense"]
        else:
            return None
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return None


def _backward_for_view(view, payload: Any, dense_query: bool | None = None) -> dict[str, Any]:
    if payload is None:
        payload = {}
    grad_by_vidx, meta = parse_backward_payload(payload, view)
    result = store.backward(view, grad_by_vidx)
    result.update(
        {
            "view_id": view.view_id,
            "frame_id": view.frame.frame_id,
            "channels": view.frame.channels,
            "parse": meta,
        }
    )

    cells = view.frame.num_points * view.frame.channels
    wants_dense = _wants_dense(payload, dense_query)
    if wants_dense is True:
        pass  # keep dense_gradients produced by the core
    elif wants_dense is False or cells > DENSE_CELL_LIMIT:
        result.pop("dense_gradients", None)
        result["dense_gradients_available"] = True
        result["dense_cell_count"] = cells
        if wants_dense is not False:
            result["dense_gradients_auto_omitted"] = True
            result["hint"] = (
                "response is sparse; pass ?dense=true or \"dense\": true "
                "to receive the full point-aligned dense_gradients array"
            )
    return result


@router.post("/views/{view_id}/backward", tags=["backward"])
def backward_view(
    view_id: str,
    dense: bool | None = None,
    payload: Any = Body(default={}, openapi_examples=BACKWARD_EXAMPLES),
) -> dict[str, Any]:
    view = store.resolve_view(view_id)
    body = payload if payload is not None else {}
    if isinstance(body, dict) and body.get("view_id") is not None:
        try:
            other = store.resolve_view(body["view_id"])
        except APIError:
            raise
        if other.view_id != view.view_id:
            raise APIError(
                409,
                "VIEW_MISMATCH",
                f"path view {view.view_id} differs from body view {other.view_id}",
            )
    return _backward_for_view(view, body, dense)


@router.post("/backward", tags=["backward"])
def backward_by_body(
    dense: bool | None = None,
    payload: dict[str, Any] = Body(
        ...,
        openapi_examples={
            "body": {
                "summary": "Backward with view_id in body",
                "value": {
                    "view_id": 2,
                    "grad_output": [
                        {"voxel_index": 0, "grad": [1.0, 2.0]},
                    ],
                },
            }
        },
    ),
) -> dict[str, Any]:
    view_ref = payload.get("view_id", payload.get("viewId", payload.get("id")))
    if view_ref is None:
        raise APIError(422, "VIEW_REQUIRED", "view_id is required in the request body")
    view = store.resolve_view(view_ref)
    return _backward_for_view(view, payload, dense)


# --------------------------------------------------------------------------- #
# Self-checking demo
# --------------------------------------------------------------------------- #
@router.post("/demo/walkthrough", tags=["demo"])
def demo_walkthrough() -> dict[str, Any]:
    return run_walkthrough(store)


@router.get("/demo/walkthrough", tags=["demo"])
def demo_walkthrough_get() -> dict[str, Any]:
    return run_walkthrough(store)


# --------------------------------------------------------------------------- #
# Hidden compatibility aliases (alternate resource naming)
# --------------------------------------------------------------------------- #
for alias in (
    "/point_clouds",
    "/point-clouds",
    "/pointcloud",
    "/point_cloud",
    "/frames",
    "/upload",
    "/pointclouds/upload",
    "/point_cloud/upload",
    "/voxel_max_pool",
    "/voxel-max-pool",
    "/maxpool",
    "/pool",
):
    router.add_api_route(
        alias, create_pointcloud, methods=["POST"], include_in_schema=False
    )

for alias in ("/frames", "/point_clouds", "/point-clouds", "/pointcloud"):
    router.add_api_route(
        alias, list_pointclouds, methods=["GET"], include_in_schema=False
    )

# Canonical router at root plus common API prefixes (hidden duplicates in docs).
app.include_router(router)
app.include_router(router, prefix="/api", include_in_schema=False)
app.include_router(router, prefix="/api/v1", include_in_schema=False)
