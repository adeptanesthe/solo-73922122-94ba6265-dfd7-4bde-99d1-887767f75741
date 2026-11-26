"""Self-checking end-to-end narrative used by ``POST /demo/walkthrough``.

The same scenario is exercised by the pytest suite and by ``scripts/demo.py``
over real HTTP.  It demonstrates masking, restoring, branching and backward
on an *old* view.
"""

from __future__ import annotations

from typing import Any

from .core import APIError, Store
from .responses import view_output


DEMO_IDS = [1, 2, 3, 4, 5]
DEMO_COORDS = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
]
DEMO_FEATURES = [
    [10.0, 0.0],   # id 1: ties channel 0 with id 2 -> smallest id wins ch0
    [10.0, 7.0],   # id 2: wins channel 1 initially, both after masking id 1
    [3.0, 5.0],    # id 3
    [2.0, 9.0],    # id 4: wins channel 1 of voxel 1
    [8.0, 1.0],    # id 5: wins channel 0 of voxel 1
]


def run_walkthrough(store: Store | None = None) -> dict[str, Any]:
    from .core import store as default_store

    st = store or default_store
    checks: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    def check(name: str, condition: bool, expected: Any, actual: Any) -> bool:
        checks.append(
            {
                "name": name,
                "passed": bool(condition),
                "expected": expected,
                "actual": actual,
            }
        )
        return bool(condition)

    # 1. Upload -------------------------------------------------------------
    frame, root = st.create_frame(DEMO_IDS, DEMO_COORDS, DEMO_FEATURES)
    out0 = view_output(root)
    w0 = [v["winner_point_ids"] for v in out0["voxels"]]
    check("initial voxel0 winners are [1,2] (tie -> id 1 on ch0)",
          w0[0] == [1, 2], [1, 2], w0[0])
    check("initial voxel1 winners are [5,4]", w0[1] == [5, 4], [5, 4], w0[1])
    steps.append({
        "step": 1,
        "action": "upload_pointcloud",
        "frame_id": frame.frame_id,
        "view_id": root.view_id,
        "output": _brief(out0),
    })

    # 2. Mask point id 1 from initial view ----------------------------------
    masked = st.apply_edit(root, mask_raw=[1], note="mask tie-winner id 1")
    out_m = view_output(masked)
    wm = [v["winner_point_ids"] for v in out_m["voxels"]]
    check("after masking id1, voxel0 winners are [2,2]",
          wm[0] == [2, 2], [2, 2], wm[0])
    check("mask edit recomputes exactly 1 voxel",
          masked.recomputed_voxels == 1, 1, masked.recomputed_voxels)
    check("mask edit has exactly 1 changed point",
          masked.changed_points == 1, 1, masked.changed_points)
    check("delta stores 1*C=2 max slots",
          masked.delta_stats()["stored_max_values"] == 2,
          2, masked.delta_stats()["stored_max_values"])
    steps.append({
        "step": 2,
        "action": "mask",
        "base_view_id": root.view_id,
        "view_id": masked.view_id,
        "request": {"point_ids": [1]},
        "recomputed_voxels": masked.recomputed_voxels,
        "changed_points": masked.changed_points,
        "output": _brief(out_m),
    })

    # 3. Restore point id 1 on top of the masked view -----------------------
    restored = st.apply_edit(masked, restore_raw=[1], note="restore id 1")
    out_r = view_output(restored)
    wr = [v["winner_point_ids"] for v in out_r["voxels"]]
    check("after restoring id1, winners return to [1,2]",
          wr[0] == [1, 2], [1, 2], wr[0])
    check("restore creates a new view id",
          restored.view_id != masked.view_id, "new id", restored.view_id)
    steps.append({
        "step": 3,
        "action": "restore",
        "base_view_id": masked.view_id,
        "view_id": restored.view_id,
        "request": {"point_ids": [1]},
        "output": _brief(out_r),
    })

    # 4. Branch: from the masked view, also mask id 2 -----------------------
    branched = st.apply_edit(masked, mask_raw=[2], note="branch: mask id2 too")
    out_b = view_output(branched)
    wb = out_b["voxels"][0]["winner_point_ids"]
    check("branch view voxel0 winners are [3,3]", wb == [3, 3], [3, 3], wb)
    steps.append({
        "step": 4,
        "action": "mask_on_masked_view",
        "base_view_id": masked.view_id,
        "view_id": branched.view_id,
        "request": {"point_ids": [2]},
        "output": _brief(out_b),
    })

    # 5. Backward on the OLD masked view (must still use [2,2] winners) -----
    grads = {
        0: __import__("numpy").asarray([1.0, 2.0]),
        1: __import__("numpy").asarray([3.0, 4.0]),
    }
    bw_old = st.backward(masked, grads)
    dense_old = bw_old["dense_gradients"]
    expected_old = [
        [0.0, 0.0],  # id 1 hidden in this view
        [1.0, 2.0],  # wins both channels of voxel 0 in old masked view
        [0.0, 0.0],
        [0.0, 4.0],  # voxel1 ch1 winner id4
        [3.0, 0.0],  # voxel1 ch0 winner id5
    ]
    check("old-view backward routes by old winners",
          dense_old == expected_old, expected_old, dense_old)
    check("old view unaffected by newer branch",
          view_output(masked)["voxels"][0]["winner_point_ids"] == [2, 2],
          [2, 2],
          view_output(masked)["voxels"][0]["winner_point_ids"])
    steps.append({
        "step": 5,
        "action": "backward_on_old_view",
        "view_id": masked.view_id,
        "upstream": [
            {"voxel_index": 0, "grad": [1.0, 2.0]},
            {"voxel_index": 1, "grad": [3.0, 4.0]},
        ],
        "dense_gradients_point_order": list(DEMO_IDS),
        "result": {
            "point_gradients": bw_old["point_gradients"],
            "sparse_gradients": bw_old["sparse_gradients"],
            "dense_gradients": bw_old["dense_gradients"],
        },
    })

    # 6. Backward on the INITIAL view shows tie routing to smallest id ------
    bw_init = st.backward(root, {0: __import__("numpy").asarray([10.0, 20.0])})
    expected_init = [
        [10.0, 0.0],   # id1 wins tied ch0 (smallest id)
        [0.0, 20.0],   # id2 wins ch1
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    check("initial-view backward splits tied channel to smallest id",
          bw_init["dense_gradients"] == expected_init,
          expected_init, bw_init["dense_gradients"])
    steps.append({
        "step": 6,
        "action": "backward_on_initial_view",
        "view_id": root.view_id,
        "upstream": [{"voxel_index": 0, "grad": [10.0, 20.0]}],
        "result": {
            "point_gradients": bw_init["point_gradients"],
            "dense_gradients": bw_init["dense_gradients"],
        },
    })

    # 7. Unknown id rejects the whole edit without creating a view ----------
    views_before = len(st.views)
    rejected_body: dict[str, Any] = {}
    try:
        st.apply_edit(root, mask_raw=[1, 999, 1000])
        rejected = False
    except APIError as exc:
        rejected = exc.code == "UNKNOWN_POINT_IDS"
        rejected_body = {"status_code": exc.status_code, "code": exc.code, "details": exc.details}
    views_after = len(st.views)
    check("edit with unknown point ids is rejected", rejected, True, rejected_body)
    check("rejected edit creates no view",
          views_before == views_after, views_before, views_after)
    steps.append({
        "step": 7,
        "action": "edit_with_unknown_ids",
        "request": {"point_ids": [1, 999, 1000]},
        "rejection": rejected_body,
        "views_before": views_before,
        "views_after": views_after,
    })

    all_passed = all(c["passed"] for c in checks)
    return {
        "all_checks_passed": all_passed,
        "check_count": len(checks),
        "passed_count": sum(c["passed"] for c in checks),
        "checks": checks,
        "steps": steps,
        "hint": "Open /docs and call POST /demo/walkthrough to rerun this scenario.",
    }


def _brief(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_id": output["view_id"],
        "parent_view_id": output["parent_view_id"],
        "changed_points": output["changed_points"],
        "recomputed_voxels": output["recomputed_voxels"],
        "recomputed_channels": output["recomputed_channels"],
        "delta": output["delta"],
        "voxels": output["voxels"],
    }
