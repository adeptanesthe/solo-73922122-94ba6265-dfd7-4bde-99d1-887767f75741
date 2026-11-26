#!/usr/bin/env python3
"""End-to-end HTTP demo for the voxel max-pool occlusion-view API.

Usage:
    python scripts/demo.py [base_url]

Defaults to http://localhost:8000 .  Only the Python standard library is
required on the host.  It demonstrates:

    1. uploading a cloud (initial view, all points visible),
    2. masking a point -> new view id,
    3. restoring it -> another new view id,
    4. branching and running backward on an OLD view,
    5. atomic rejection of edits containing unknown point ids.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
BASE = BASE.rstrip("/")

PASS = 0
FAIL = 0


def call(method: str, path: str, body=None, expect_error: bool = False):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        if expect_error:
            return exc.code, payload
        raise SystemExit(f"HTTP {exc.code} on {method} {path}: {payload}")


def check(name: str, ok: bool, expected=None, actual=None) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  \033[32mPASS\033[0m {name}")
    else:
        FAIL += 1
        print(f"  \033[31mFAIL\033[0m {name}\n         expected={expected!r}\n         actual={actual!r}")


def brief_voxels(doc):
    return [
        {"coord": v["voxel_coord"], "max": v["max_values"], "winners": v["winner_point_ids"]}
        for v in doc.get("voxels", [])
    ]


def main() -> int:
    print(f"== Voxel MaxPool demo against {BASE} ==")
    status, health = call("GET", "/health")
    check("service healthy", status == 200 and health["status"] == "ok", 200, status)

    # 1. Upload -------------------------------------------------------------
    print("\n[1] Upload point cloud (initial view: all visible)")
    upload = {
        "point_ids": [1, 2, 3, 4, 5],
        "voxel_coords": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
        "features": [[10, 0], [10, 7], [3, 5], [2, 9], [8, 1]],
    }
    _, up = call("POST", "/pointclouds", upload)
    frame_id = up["frame_id"]
    v_initial = up["view_id"]
    print("   frame_id =", frame_id, " initial view_id =", v_initial)
    print("   voxels   =", json.dumps(brief_voxels(up)))
    check("initial tie ch0 -> smallest id 1",
          up["voxels"][0]["winner_point_ids"] == [1, 2], [1, 2],
          up["voxels"][0]["winner_point_ids"])

    # 2. Mask ---------------------------------------------------------------
    print("\n[2] Mask point 1 on the initial view -> new view")
    _, masked = call("POST", f"/views/{v_initial}/mask", {"point_ids": [1]})
    v_masked = masked["view_id"]
    print("   new view_id =", v_masked,
          "| changed =", masked["changed_points"],
          "| recomputed voxels =", masked["recomputed_voxels"],
          "| channels =", masked["recomputed_channels"])
    print("   voxels =", json.dumps(brief_voxels(masked)))
    check("mask flips voxel0 winners to [2,2]",
          masked["voxels"][0]["winner_point_ids"] == [2, 2])
    check("exactly one voxel recomputed", masked["recomputed_voxels"] == 1)

    # 3. Restore ------------------------------------------------------------
    print("\n[3] Restore point 1 on the masked view -> another new view")
    _, restored = call("POST", f"/views/{v_masked}/restore", {"point_ids": [1]})
    v_restored = restored["view_id"]
    print("   new view_id =", v_restored)
    print("   voxels =", json.dumps(brief_voxels(restored)))
    check("restore returns winners to [1,2]",
          restored["voxels"][0]["winner_point_ids"] == [1, 2])
    check("restore yields a distinct new view id",
          v_restored not in (v_initial, v_masked))

    # 4a. Branch away from the masked view --------------------------------
    print("\n[4a] Branch: also mask point 2 on the OLD masked view")
    _, branched = call("POST", f"/views/{v_masked}/mask", {"point_ids": [2]})
    v_branch = branched["view_id"]
    print("   branch view_id =", v_branch,
          "| voxel0 winners =", branched["voxels"][0]["winner_point_ids"])
    check("branch winners become [3,3]",
          branched["voxels"][0]["winner_point_ids"] == [3, 3])

    # 4b. Backward on the OLD masked view ----------------------------------
    print("\n[4b] Backward on OLD masked view", v_masked,
          "(must still use its [2,2] winners)")
    bw_body = {
        "grad_output": [
            {"voxel_coord": [0, 0, 0], "grad": [1.0, 2.0]},
            {"voxel_coord": [1, 0, 0], "grad": [3.0, 4.0]},
        ]
    }
    _, bw = call("POST", f"/views/{v_masked}/backward", bw_body)
    expected_dense = [
        [0.0, 0.0],
        [1.0, 2.0],
        [0.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
    ]
    print("   dense (point order 1..5) =", json.dumps(bw["dense_gradients"]))
    print("   sparse =", json.dumps(bw["sparse_gradients"]))
    check("old-view backward matches winning ids at that time",
          bw["dense_gradients"] == expected_dense, expected_dense, bw["dense_gradients"])

    # Initial-view backward shows tie routing to smallest id.
    _, bw0 = call(
        "POST",
        f"/views/{v_initial}/backward",
        {"grad_output": [{"voxel_index": 0, "grad": [10.0, 20.0]}]},
    )
    check("initial-view tie routes ch0 grad to id 1",
          bw0["gradients_by_point"].get("1") == [10.0, 0.0],
          [10.0, 0.0], bw0["gradients_by_point"].get("1"))

    # 5. Unknown ids -> whole edit rejected --------------------------------
    print("\n[5] Edit with unknown point id 999 -> whole request rejected")
    status, err = call(
        "POST", f"/views/{v_initial}/mask",
        {"point_ids": [2, 999]}, expect_error=True,
    )
    print("  ", status, err.get("code"), err.get("details"))
    check("unknown-id edit returns 400 UNKNOWN_POINT_IDS",
          status == 400 and err.get("code") == "UNKNOWN_POINT_IDS")
    _, still = call("GET", f"/views/{v_initial}/output")
    check("initial view unchanged after rejected edit",
          still["voxels"][0]["winner_point_ids"] == [1, 2])

    print(f"\nResult: {PASS} passed, {FAIL} failed")
    print("Explore interactively at", BASE + "/docs")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
