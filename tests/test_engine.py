"""Core engine tests: tie-breaking, deltas, incremental recompute, backward."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "/workspace")

from app.core import APIError, Store, grouped_max_pool  # noqa: E402


IDS = [1, 2, 3, 4, 5]
COORDS = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
]
FEATS = [
    [10.0, 0.0],
    [10.0, 7.0],
    [3.0, 5.0],
    [2.0, 9.0],
    [8.0, 1.0],
]


@pytest.fixture()
def store():
    return Store()


@pytest.fixture()
def cloud(store):
    frame, root = store.create_frame(IDS, COORDS, FEATS)
    return store, frame, root


def winners(view, vidx=0):
    return [view.frame.point_ids[int(p)] if p >= 0 else None
            for p in view.get_voxel_state(vidx)[1]]


def test_initial_pool_values_and_tie_break(cloud):
    store, frame, root = cloud
    v0, w0, _ = root.get_voxel_state(0)
    assert v0.tolist() == [10.0, 7.0]
    assert [frame.point_ids[int(p)] for p in w0] == [1, 2]  # tie ch0 -> id1
    v1, w1, _ = root.get_voxel_state(1)
    assert v1.tolist() == [8.0, 9.0]
    assert [frame.point_ids[int(p)] for p in w1] == [5, 4]


def test_mask_single_point_recomputes_one_voxel_and_small_delta(cloud):
    store, frame, root = cloud
    child = store.apply_edit(root, mask_raw=[1])
    assert child.changed_points == 1
    assert child.recomputed_voxels == 1
    assert child.recomputed_channels == 2
    assert child.dirty_vidx.tolist() == [0]
    # Own arrays are exactly [1, C], not [V, C] or [N, C].
    assert child.values.shape == (1, 2)
    assert child.winner_pidx.shape == (1, 2)
    assert len(child.vis_delta) == 1
    assert winners(child, 0) == [2, 2]
    # Clean voxel inherited unchanged through the chain.
    assert winners(child, 1) == [5, 4]


def test_mask_already_masked_is_noop(cloud):
    store, frame, root = cloud
    v2 = store.apply_edit(root, mask_raw=[1])
    v3 = store.apply_edit(v2, mask_raw=[1, 1])  # duplicate + already hidden
    assert v3.changed_points == 0
    assert v3.recomputed_voxels == 0
    assert v3.dirty_vidx.shape == (0,)
    assert v3.values.shape == (0, 2)
    # State still correct.
    assert winners(v3, 0) == [2, 2]


def test_restore_flips_back(cloud):
    store, frame, root = cloud
    v2 = store.apply_edit(root, mask_raw=[1, 2])
    assert winners(v2, 0) == [3, 3]
    v3 = store.apply_edit(v2, restore_raw=[1])
    # id1 visible again; ch0 tie 10 between id1 and (still hidden) id2 -> id1.
    # ch1: id3=5 vs id1=0 -> id3.
    assert winners(v3, 0) == [1, 3]
    v4 = store.apply_edit(v3, restore_raw=[2])
    assert winners(v4, 0) == [1, 2]


def test_unknown_ids_reject_entire_edit(cloud):
    store, frame, root = cloud
    before_views = len(store.views)
    before_edits = store.edit_count
    with pytest.raises(APIError) as ei:
        store.apply_edit(root, mask_raw=[1, 424242], restore_raw=[999])
    assert ei.value.code == "UNKNOWN_POINT_IDS"
    unknown = ei.value.details["unknown_point_ids"]
    assert set(unknown) == {424242, 999}
    assert len(store.views) == before_views
    assert store.edit_count == before_edits
    # Root untouched.
    assert winners(root, 0) == [1, 2]


def test_branching_keeps_old_view_winners(cloud):
    store, frame, root = cloud
    masked = store.apply_edit(root, mask_raw=[1])
    # Branch A: mask 2 on top of masked.
    branch = store.apply_edit(masked, mask_raw=[2])
    assert winners(branch, 0) == [3, 3]
    # Old masked view must be unchanged.
    assert winners(masked, 0) == [2, 2]
    assert winners(root, 0) == [1, 2]

    # Backward on old masked view routes to its winners.
    res = store.backward(masked, {0: np.asarray([1.0, 2.0])})
    dense = res["dense_gradients"]
    assert dense[1] == [1.0, 2.0]  # pidx 1 == point id 2
    assert dense[0] == [0.0, 0.0]  # point id 1 hidden
    assert dense[2] == [0.0, 0.0]


def test_backward_per_channel_split_and_sparse(cloud):
    store, frame, root = cloud
    res = store.backward(
        root,
        {0: np.asarray([10.0, 20.0]), 1: np.asarray([30.0, 40.0])},
    )
    dense = res["dense_gradients"]
    # id1 wins ch0 of voxel0 (tie), id2 wins ch1.
    assert dense[0] == [10.0, 0.0]
    assert dense[1] == [0.0, 20.0]
    # voxel1 winners [id5 (ch0), id4 (ch1)]
    assert dense[4] == [30.0, 0.0]
    assert dense[3] == [0.0, 40.0]
    assert dense[2] == [0.0, 0.0]

    by_point = res["gradients_by_point"]
    assert by_point["1"] == [10.0, 0.0]
    assert by_point["2"] == [0.0, 20.0]
    sparse_pairs = {(e["point_id"], e["channel"]): e["value"]
                    for e in res["sparse_gradients"]}
    assert sparse_pairs[(1, 0)] == 10.0
    assert sparse_pairs[(2, 1)] == 20.0
    assert sparse_pairs[(5, 0)] == 30.0
    assert sparse_pairs[(4, 1)] == 40.0
    assert res["routed_channels"] == 4


def test_mask_all_points_empties_voxel(cloud):
    store, frame, root = cloud
    child = store.apply_edit(root, mask_raw=[1, 2, 3])
    vals, win, src = child.get_voxel_state(0)
    assert win[0] == -1 and win[1] == -1
    # Voxel 1 still populated and inherited.
    assert child.get_voxel_state(1)[1][0] >= 0


def test_only_dirty_voxels_recomputed_with_spy(cloud, monkeypatch):
    import app.core as core

    store, frame, root = cloud
    calls = []
    orig = core.grouped_max_pool

    def spy(features, group_ids, num_groups, visible):
        calls.append(num_groups)
        return orig(features, group_ids, num_groups, visible)

    monkeypatch.setattr(core, "grouped_max_pool", spy)
    store.apply_edit(root, mask_raw=[1])
    assert calls[-1] == 1  # only one dirty voxel pooled
    store.apply_edit(root, mask_raw=[1, 4])
    assert calls[-1] == 2  # two dirty voxels pooled


def test_delta_storage_scales_with_changes_not_cloud(store):
    # Big cloud: one voxel with many points + many singleton voxels.
    n = 2000
    ids = list(range(n))
    coords = [[0, 0, 0]] * n  # all same voxel, C=8
    feats = np.random.RandomState(0).randn(n, 8).tolist()
    frame, root = store.create_frame(ids, coords, feats)
    assert frame.num_voxels == 1
    child = store.apply_edit(root, mask_raw=[5])
    stats = child.delta_stats()
    assert stats["visibility_entries"] == 1
    assert stats["voxel_result_entries"] == 1
    assert stats["stored_max_values"] == 8
    # The child's own numpy resident delta is tiny vs the full feature matrix.
    assert child.values.nbytes + child.winner_pidx.nbytes <= 8 * 8 * 2 + 64


def test_many_views_chain_resolution(cloud):
    store, frame, root = cloud
    cur = root
    # Alternate masking/restoring id 1.
    for i in range(10):
        if i % 2 == 0:
            cur = store.apply_edit(cur, mask_raw=[1])
        else:
            cur = store.apply_edit(cur, restore_raw=[1])
    # Last iteration i=9 odd -> restore -> id1 winner again.
    assert winners(cur, 0) == [1, 2]
    # Root immutable.
    assert winners(root, 0) == [1, 2]


def test_duplicate_point_ids_rejected(store):
    with pytest.raises(APIError) as ei:
        store.create_frame([1, 1], [[0], [0]], [[1.0], [2.0]])
    assert ei.value.code == "DUPLICATE_POINT_ID"


def test_dimension_mismatches_rejected(store):
    with pytest.raises(APIError):
        store.create_frame([1, 2], [[0, 0]], [[1.0]])
    with pytest.raises(APIError):
        store.create_frame([1, 2], [[0], [1]], [[1.0, 2.0], [1.0]])


def test_grouped_pool_empty_input():
    vals, rows = grouped_max_pool(
        np.zeros((0, 3)), np.zeros(0, dtype=np.int64), 0, np.ones(0, dtype=bool)
    )
    assert vals.shape == (0, 3)
    assert rows.shape == (0, 3)


def test_view_does_not_retain_large_noop_request(cloud):
    store, frame, root = cloud
    once = store.apply_edit(root, mask_raw=[1])
    # Huge request, but every id is already hidden -> zero actual changes.
    huge = [1] * 100000
    child = store.apply_edit(once, mask_raw=huge)
    assert child.changed_points == 0
    assert child.vis_delta == {}
    # Request payloads must not become resident view state.
    assert not hasattr(child, "mask_requested")
    assert not hasattr(child, "restore_requested")
    # The view's own arrays carry no voxel slots when nothing changed.
    assert child.values.nbytes == 0
    assert child.winner_pidx.nbytes == 0
