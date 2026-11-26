# Voxel MaxPool Occlusion-View API

FastAPI + NumPy service for the occlusion-view workflow in point-cloud
segmentation training. One uploaded frame derives many immutable **views**
(masking / restoring batches of points). Each edit returns a new view id and
the per-channel maxima of every non-empty voxel. Backward on *any* view routes
upstream voxel gradients to the point ids that were winning **at that view's
time**, with deterministic smallest-id tie-breaking.

## Implemented behavior (requirement map)

| Requirement | Implementation |
| --- | --- |
| Unique point ids, integer voxel coords, equal-length features | `POST /pointclouds`; strict validation (`DUPLICATE_POINT_ID`, `NON_INTEGER_COORD`, `FEATURE_DIM_MISMATCH`, …). Columnar and row-object JSON shapes accepted. |
| Initial view: all points visible | Root view `view_id` returned on upload; empty `vis_delta` means default-visible. |
| Batch mask / restore on **any** view → new view id + all non-empty voxel maxima | `POST /views/{view_id}/mask`, `/restore`, `/edit` (also frame-scoped `…/pointclouds/{fid}/views/{ref}/…`, body-level `/mask`, `/restore`, `/edit`). Response includes `view_id` and full `voxels` / `max_values` / `winner_point_ids`. |
| Ties → smallest point id per channel | Members of each voxel are stored ascending by point id; first-`argmax` therefore selects the smallest id on exact ties. Verified by randomized brute-force tests. |
| Recompute only voxels of actually-changed points | Actual flips are computed against the base view; `dirty_vidx = unique(voxel(p) for p in changed)`. Already-hidden masks / already-visible restores are no-ops and recompute nothing. |
| New-view resident state grows only with changed points + those voxels' channels | A view stores only `vis_delta` (`K`), `changed_pidx` (`K`), and `values`/`winner_pidx` arrays of shape `(D, C)` plus a `(D,)` index. It does **not** retain request id lists. Exposed as `delta` / `resident_delta` in every view response. |
| Backward by view id + upstream gradients of non-empty voxels → sparse point gradients by then-winners | `POST /views/{view_id}/backward` and `POST /backward`. Winners are resolved from that immutable view's delta chain, so old views keep their old routing. Returns `sparse_gradients`, COO `sparse`, `point_gradients`, `gradients_by_point`, and a point-aligned `dense_gradients` for small clouds (auto-sparse at >65 536 cells; `?dense=true` to force). |
| Unknown point id in an edit → reject whole request | All ids are resolved first; any unknown id yields `400 UNKNOWN_POINT_IDS` with the full list, no view allocated/registered, no state change. |
| Start with Docker Compose; demonstrate mask / restore / old-view backward from `/docs` | `docker compose up --build` → <http://localhost:8000/docs>. Every endpoint ships prefilled examples; `POST /demo/walkthrough` runs and self-checks the whole story in one call; `scripts/demo.py` drives it over HTTP. |

### Delta-chain design

```
Frame (immutable): ids, int coords, [N,C] features, voxel grouping,
                   per-voxel member indices sorted by point id
View root  (depth 0): results for ALL voxels (baseline resident state)
View child (depth k): parent pointer
                      vis_delta   {pidx: visible}   # K actual flips
                      dirty_vidx  [D]
                      values       [D,C] float64     # recomputed maxima
                      winner_pidx  [D,C] int64 (-1 = empty voxel)
```

Visibility and voxel winners are resolved nearest-ancestor-wins while walking
the parent chain. Children never mutate parents, which is why backward on an
old view stays correct after arbitrary branching.

## Quick start

### Docker Compose

```bash
docker compose up --build
# then open:
#   http://localhost:8000/docs      (Swagger UI)
#   http://localhost:8000/health
```

Host port `8000` can be overridden with `APP_PORT=8080 docker compose up`.

Optional host-side HTTP demo (Python stdlib only):

```bash
python scripts/demo.py http://localhost:8000
# or inside the container:
docker compose exec voxel-pool-api python scripts/demo.py
```

### Local virtualenv (without Docker)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest -q
```

## Demonstrating in `/docs` (click-by-click)

All request bodies already contain working examples — choose an example in
each endpoint and press **Execute**.

1. **One-call self-check:** open `POST /demo/walkthrough` → **Execute**.
   It uploads the demo cloud, masks, restores, branches, performs backward on
   the old masked view, and checks atomic unknown-id rejection
   (`"all_checks_passed": true`).
2. **Upload:** `POST /pointclouds` → Execute. Note `frame_id` and
   `view_id` (the initial all-visible view, e.g. `1`). Voxel 0 winners are
   `[1, 2]` — channel 0 is a `10 == 10` tie between ids `1` and `2`, so the
   smallest id (`1`) wins.
3. **Mask:** `POST /views/{view_id}/mask` (use the returned id) with
   `{"point_ids":[1]}`. The response has a **new** `view_id`; voxel 0 winners
   become `[2, 2]`, `recomputed_voxels = 1`, and `delta.stored_max_values = 2`
   (`D*C`).
4. **Restore:** `POST /views/{view_id}/restore` using the masked view's id
   with `{"point_ids":[1]}` → another new id; winners return to `[1, 2]`.
5. **Old-view backward:** branch once more from the masked view
   (`mask [2]`), then call `POST /views/{masked_view_id}/backward` with
   ```json
   {"grad_output": [
     {"voxel_index": 0, "grad": [1.0, 2.0]},
     {"voxel_index": 1, "grad": [3.0, 4.0]}
   ]}
   ```
   Even though the newer branch changed voxel 0 to winners `[3,3]`, the old
   masked view still routes both channels of voxel 0 to point `2`:
   `dense_gradients == [[0,0],[1,2],[0,0],[0,4],[3,0]]` (point order 1…5).

### Backward payload formats accepted

* `{"grad_output":[{"voxel_coord":[0,0,0],"grad":[…]}, …]}`
* same with `"voxel_index"` / `"voxel_id"`
* columnar `{"voxel_coords":[…], "grad_values":[…]}`
* positional `{"grad_values":[[…], …]}` aligned to the response's non-empty
  voxel order (or all `frame.num_voxels` frame voxels, empty ones skipped)
* body-level `POST /backward` with `"view_id"`
* mapping `{"gradients_by_voxel": {"0,0,0": […]}}`

Unknown / currently-empty voxel references are ignored by default and listed
under `parse`; pass `"strict": true` to reject them with 4xx.

## API surface

| Method & path | Purpose |
| --- | --- |
| `POST /pointclouds` (`/point_clouds`, `/frames`, `/upload`, `/api/v1/…`) | Upload frame + initial view |
| `GET /pointclouds`, `GET /pointclouds/{frame_id}` | List / frame detail with voxel membership |
| `GET /views`, `GET /views/{view_id}`, `GET /views/{view_id}/output` | View metadata + resolved forward output |
| `POST /views/{view_id}/mask` · `/restore` · `/edit` | Delta edits → new view + full output |
| `POST /pointclouds/{fid}/views/{initial|latest|id}/mask|restore` | Frame-scoped edits (`initial` alias) |
| `POST /mask` · `/restore` · `/edit` | Same, with `view_id` in the JSON body |
| `POST /views/{view_id}/backward` · `POST /backward` | Sparse / dense point gradients |
| `POST|GET /demo/walkthrough` | Self-checking scenario |
| `GET /health` · `GET /stats` | Liveness and cumulative recompute counters |

View references accept `3`, `"3"`, `"v3"`, `"view-3"`, `"view_003"`.

## Project layout

```
app/core.py      # Frame/View/Store engine, NumPy pooling, delta resolution
app/parsing.py   # tolerant request parsing (multiple JSON shapes)
app/responses.py # JSON serialization
app/demo.py      # deterministic self-checking walkthrough
app/main.py      # FastAPI routes + OpenAPI examples
scripts/demo.py  # stdlib-only HTTP demo for docker compose
tests/           # 37 engine + HTTP tests
Dockerfile docker-compose.yml requirements*.txt
```

## Verification results

Executed in the delivery environment with Python 3.11 and the pinned
dependencies installed into an isolated prefix (no Docker daemon/CLI exists in
this build sandbox, so the image itself was not built here):

* `pytest -q` → **37 passed**.
* Live `uvicorn` server (started with only `requirements.txt` on
  `PYTHONPATH`, i.e. the image's runtime dependency set):
  `python scripts/demo.py` → **11/11 checks passed**.
* Replayed **all 17 request examples embedded in `/docs`** over real HTTP →
  all `200`.
* Randomized differential test vs. a brute-force Python implementation
  (300 points, 63 voxels, 5 channels): all channel maxima/tie winners and a
  15-op chained-view backward matched exactly.
* Scale sanity: 100 000 points / ~1 000 voxels / 16 channels upload+root pool
  ≈ 1.7 s; chained single-point edits ≈ 1 ms/edit; 20 concurrent branching
  edits produced unique view ids.
* `docker-compose.yml` parses as valid YAML; the image `HEALTHCHECK` one-liner
  was executed successfully against a live server;
  `pip install -r requirements.txt` into a clean target succeeded.

## Known limitations / unfinished items

* **Docker Compose was not actually launched here** — this sandbox has neither
  the `docker` CLI nor a daemon socket. The Compose file, Dockerfile and
  healthcheck are provided and statically/dependency verified;
  `docker compose up --build` must be confirmed on a Docker-capable host.
* Storage is **in-process memory only**; there is no persistence,
  authentication or multi-tenancy (frame ids are sequential). Restarting the
  container loses state. This matches the interactive training-session scope.
* Delta-chain resolution cost grows with lineage depth for full-output
  assembly; very long chains could be accelerated with periodic snapshots
  (not needed for correctness, and per-edit recompute itself stays O(dirty)).
* The full dense `N×C` backward matrix is omitted by default for large clouds;
  force it with `?dense=true` if your client wants point-aligned dense grads.
* Coordinates are integer lists (any dimension ≥ 1); dict-style `{x,y,z}`
  coordinates are not parsed.
