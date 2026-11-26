"""End-to-end HTTP tests through the FastAPI TestClient."""

import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "/workspace")

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


UPLOAD_BODY = {
    "point_ids": [1, 2, 3, 4, 5],
    "voxel_coords": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
    "features": [[10, 0], [10, 7], [3, 5], [2, 9], [8, 1]],
}


def upload(client, body=None):
    r = client.post("/pointclouds", json=body or UPLOAD_BODY)
    assert r.status_code == 200, r.text
    return r.json()


def test_docs_and_health(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_prefix_aliases(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/v1/health").status_code == 200
    r = client.post("/api/v1/pointclouds", json=UPLOAD_BODY)
    assert r.status_code == 200, r.text
    r2 = client.post("/frames", json=UPLOAD_BODY)
    assert r2.status_code == 200, r2.text


def test_upload_initial_output(client):
    j = upload(client)
    assert j["point_count"] == 5
    assert j["channels"] == 2
    assert j["initial_view_id"] == j["view_id"]
    assert [v["winner_point_ids"] for v in j["voxels"]] == [[1, 2], [5, 4]]
    assert j["max_values"] == [[10.0, 7.0], [8.0, 9.0]]


def test_row_format_upload(client):
    body = {
        "points": [
            {"id": 7, "coord": [2, 2, 2], "feature_vector": [1.0, 2.0]},
            {"id": 8, "coord": [2, 2, 2], "feature_vector": [3.0, 0.0]},
        ]
    }
    j = upload(client, body)
    assert j["point_count"] == 2
    assert j["voxels"][0]["winner_point_ids"] == [8, 7]


def test_mask_restore_and_new_ids(client):
    j0 = upload(client)
    vid0 = j0["view_id"]

    r = client.post(f"/views/{vid0}/mask", json={"point_ids": [1]})
    assert r.status_code == 200, r.text
    jm = r.json()
    vidm = jm["view_id"]
    assert vidm != vid0
    assert jm["parent_view_id"] == vid0
    assert jm["changed_points"] == 1
    assert jm["recomputed_voxels"] == 1
    assert jm["voxels"][0]["winner_point_ids"] == [2, 2]

    # String view ref aliases work.
    r2 = client.post(f"/views/v{vidm}/restore", json={"point_ids": [1]})
    assert r2.status_code == 200
    jr = r2.json()
    assert jr["view_id"] != vidm
    assert jr["voxels"][0]["winner_point_ids"] == [1, 2]


def test_bare_list_mask(client):
    j0 = upload(client)
    r = client.post(f"/views/{j0['view_id']}/mask", json=[1, 2])
    assert r.status_code == 200, r.text
    assert r.json()["voxels"][0]["winner_point_ids"] == [3, 3]


def test_unknown_id_edit_rejected_atomically(client):
    j0 = upload(client)
    vid0 = j0["view_id"]
    before = client.get("/stats").json()["views"]

    r = client.post(f"/views/{vid0}/mask", json={"point_ids": [1, 99999]})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "UNKNOWN_POINT_IDS"
    assert 99999 in body["details"]["unknown_point_ids"]

    after = client.get("/stats").json()["views"]
    assert after == before  # no new view created

    # Existing view unchanged.
    out = client.get(f"/views/{vid0}/output").json()
    assert out["voxels"][0]["winner_point_ids"] == [1, 2]


def test_unknown_view_404(client):
    r = client.post("/views/99999999/mask", json={"point_ids": [1]})
    assert r.status_code == 404


def test_backward_shapes_and_old_view(client):
    j0 = upload(client)
    vid0 = j0["view_id"]
    vidm = client.post(f"/views/{vid0}/mask", json={"point_ids": [1]}).json()["view_id"]
    # Branch further away.
    client.post(f"/views/{vidm}/mask", json={"point_ids": [2]})

    # Backward on the OLD masked view via coordinate items.
    body = {
        "grad_output": [
            {"voxel_coord": [0, 0, 0], "grad": [1.0, 2.0]},
            {"voxel_index": 1, "grad": [3.0, 4.0]},
        ]
    }
    r = client.post(f"/views/{vidm}/backward", json=body)
    assert r.status_code == 200, r.text
    jb = r.json()
    assert jb["dense_gradients"] == [
        [0.0, 0.0],
        [1.0, 2.0],
        [0.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
    ]
    assert jb["gradients_by_point"]["2"] == [1.0, 2.0]
    assert {(e["point_id"], e["channel"]) for e in jb["sparse_gradients"]} == {
        (2, 0), (2, 1), (4, 1), (5, 0)
    }

    # Old masked view winners must still be [2,2].
    out = client.get(f"/views/{vidm}").json()
    assert out["voxels"][0]["winner_point_ids"] == [2, 2]


def test_backward_columnar_and_body_view_id(client):
    j0 = upload(client)
    vid = j0["view_id"]
    r = client.post(
        "/backward",
        json={
            "view_id": vid,
            "voxel_coords": [[0, 0, 0]],
            "grad_values": [[7.0, 9.0]],
        },
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["dense_gradients"][0] == [7.0, 0.0]   # id1 ch0
    assert j["dense_gradients"][1] == [0.0, 9.0]   # id2 ch1


def test_backward_positional_matrix(client):
    j0 = upload(client)
    vid = j0["view_id"]
    r = client.post(
        f"/views/{vid}/backward",
        json={"grad_values": [[1.0, 2.0], [3.0, 4.0]]},
    )
    assert r.status_code == 200, r.text
    dense = r.json()["dense_gradients"]
    assert dense[0] == [1.0, 0.0]
    assert dense[1] == [0.0, 2.0]
    assert dense[4] == [3.0, 0.0]
    assert dense[3] == [0.0, 4.0]


def test_backward_wrong_width_rejected(client):
    j0 = upload(client)
    r = client.post(
        f"/views/{j0['view_id']}/backward",
        json={"grad_output": [{"voxel_index": 0, "grad": [1.0]}]},
    )
    assert r.status_code == 422


def test_backward_empty_voxel_lenient_and_strict(client):
    j0 = upload(client)
    vid0 = j0["view_id"]
    vid = client.post(f"/views/{vid0}/mask", json={"point_ids": [1, 2, 3]}).json()["view_id"]

    # Lenient default: ignored empty voxel reported, voxel1 still applied.
    r = client.post(
        f"/views/{vid}/backward",
        json={"grad_output": [
            {"voxel_index": 0, "grad": [1.0, 1.0]},
            {"voxel_index": 1, "grad": [5.0, 6.0]},
        ]},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["parse"]["ignored_empty_voxels"] == [[0, 0, 0]]
    assert j["dense_gradients"][4] == [5.0, 0.0]
    assert j["dense_gradients"][3] == [0.0, 6.0]

    # Strict mode rejects.
    r2 = client.post(
        f"/views/{vid}/backward",
        json={"strict": True, "grad_output": [{"voxel_index": 0, "grad": [1.0, 1.0]}]},
    )
    assert r2.status_code == 422
    assert r2.json()["code"] == "VOXEL_EMPTY"


def test_frame_scoped_initial_alias(client):
    j0 = upload(client)
    fid = j0["frame_id"]
    r = client.post(
        f"/pointclouds/{fid}/views/initial/mask", json={"point_ids": [1]}
    )
    assert r.status_code == 200, r.text
    assert r.json()["changed_points"] == 1
    r2 = client.get(f"/pointclouds/{fid}/views/initial/output")
    assert r2.status_code == 200
    assert r2.json()["voxels"][0]["winner_point_ids"] == [1, 2]


def test_combined_edit_and_operations(client):
    j0 = upload(client)
    vid = j0["view_id"]
    r = client.post(
        f"/views/{vid}/edit",
        json={"mask": [2, 3], "restore": [2]},  # overlap -> restore wins
    )
    assert r.status_code == 200, r.text
    j = r.json()
    # 3 masked, then 2 restored -> only 3 actually flips.
    assert j["changed_points"] == 1
    assert j["voxels"][0]["winner_point_ids"] == [1, 2]

    r2 = client.post(
        f"/views/{vid}/edit",
        json={"operations": [
            {"point_id": 1, "action": "mask"},
            {"point_id": 2, "action": "mask"},
        ]},
    )
    assert r2.status_code == 200
    assert r2.json()["voxels"][0]["winner_point_ids"] == [3, 3]


def test_invalid_uploads(client):
    r = client.post("/pointclouds", json={"point_ids": [1, 1],
                                          "voxel_coords": [[0], [0]],
                                          "features": [[1], [2]]})
    assert r.status_code == 422
    assert r.json()["code"] == "DUPLICATE_POINT_ID"

    r = client.post("/pointclouds", json={"point_ids": [1],
                                          "voxel_coords": [[0.5, 0]],
                                          "features": [[1, 2]]})
    assert r.status_code == 422


def test_demo_walkthrough_endpoint(client):
    r = client.post("/demo/walkthrough")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["all_checks_passed"] is True, [c for c in j["checks"] if not c["passed"]]
    assert j["passed_count"] == j["check_count"]


def test_body_level_mask_restore_edit(client):
    j0 = upload(client)
    vid = j0["view_id"]

    r = client.post("/mask", json={"view_id": vid, "point_ids": [1]})
    assert r.status_code == 200, r.text
    vm = r.json()["view_id"]
    assert r.json()["voxels"][0]["winner_point_ids"] == [2, 2]

    r2 = client.post("/restore", json={"view_id": vm, "point_ids": [1]})
    assert r2.status_code == 200
    assert r2.json()["voxels"][0]["winner_point_ids"] == [1, 2]

    r3 = client.post("/edit", json={"view_id": vid, "mask": [2]})
    assert r3.status_code == 200
    # id1 keeps tied ch0; ch1 passes from masked id2 (7) to id3 (5).
    assert r3.json()["voxels"][0]["winner_point_ids"] == [1, 3]

    # view_id is required for body-level routes.
    assert client.post("/mask", json={"point_ids": [1]}).status_code == 422


def test_upload_aliases(client):
    for path in ("/upload", "/point_clouds", "/api/v1/frames"):
        r = client.post(path, json=UPLOAD_BODY)
        assert r.status_code == 200, (path, r.text)


def test_empty_body_mask_uses_empty_batch(client):
    j0 = upload(client)
    r = client.post(f"/views/{j0['view_id']}/mask")
    assert r.status_code == 200, r.text
    assert r.json()["changed_points"] == 0


def test_get_view_and_listing(client):
    j0 = upload(client)
    vid = j0["view_id"]
    r = client.get(f"/views/{vid}")
    assert r.status_code == 200
    assert r.json()["view_id"] == vid
    assert client.get("/views").status_code == 200
    assert client.get(f"/pointclouds/{j0['frame_id']}").status_code == 200
    assert client.get("/pointclouds/missing").status_code == 404


def test_dense_gradient_gating(client, monkeypatch):
    import app.main as main_mod

    j0 = upload(client)
    vid = j0["view_id"]
    body = {"grad_output": [{"voxel_index": 0, "grad": [1.0, 2.0]}]}

    # Small cloud: dense attached by default.
    r = client.post(f"/views/{vid}/backward", json=body)
    assert "dense_gradients" in r.json()

    # Explicit opt-out.
    r = client.post(f"/views/{vid}/backward?dense=false", json=body)
    j = r.json()
    assert "dense_gradients" not in j
    assert j["dense_gradients_available"] is True
    # Sparse representation still present and correct.
    assert j["gradients_by_point"]["1"] == [1.0, 0.0]

    # Auto threshold crossed -> omitted unless explicitly requested.
    monkeypatch.setattr(main_mod, "DENSE_CELL_LIMIT", 4)
    r = client.post(f"/views/{vid}/backward", json=body)
    assert "dense_gradients" not in r.json()
    r = client.post(f"/views/{vid}/backward?dense=true", json=body)
    assert "dense_gradients" in r.json()
    r = client.post(f"/views/{vid}/backward", json={**body, "dense": True})
    assert "dense_gradients" in r.json()
