"""HTTP routes for the sneaker price estimator API.

Every handler is a thin layer: parse/validate the request, delegate to
`current_app.predictor` (a `PricePredictor`, loaded once in
`api/app.py`) and/or `api/db.py`, then shape a JSON response. Every
400/404/409 responds with `{"error": "<message>"}` at the matching
status code, via the `_error` helper below.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

from flask import Blueprint, current_app, jsonify, request

from sneakerml.api import db

bp = Blueprint("api", __name__, url_prefix="/api")


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _predictor():
    return current_app.predictor


def _display_name_for(predictor, sneaker_id: str) -> str:
    """Look up a catalog entry's display name by exact id.

    `PricePredictor.search` only does fuzzy substring search, so an exact
    lookup is done directly against the (small) catalog DataFrame it
    already holds in memory.
    """
    match = predictor.catalog.loc[predictor.catalog["id"] == sneaker_id]
    if match.empty:
        return sneaker_id
    return str(match.iloc[0]["display_name"])


def _snapshot_dict(row: sqlite3.Row) -> dict:
    return {
        "as_of": row["as_of"],
        "low": row["low"],
        "mid": row["mid"],
        "high": row["high"],
        "extrapolated": bool(row["extrapolated"]),
    }


def _watchlist_item_response(conn, predictor, item: sqlite3.Row) -> dict:
    """Shape one `GET /api/watchlist` / `POST /api/watchlist` item.

    ``latest`` is the most recent snapshot; ``history`` is every snapshot
    as a compact ``{as_of, mid}`` pair for charting.
    """
    snapshots = db.snapshots_for(conn, item["id"])
    latest_row = snapshots[-1] if snapshots else None
    return {
        "id": item["id"],
        "sneaker_id": item["sneaker_id"],
        "display_name": _display_name_for(predictor, item["sneaker_id"]),
        "size": item["size"],
        "created_at": item["created_at"],
        "latest": _snapshot_dict(latest_row) if latest_row else None,
        "history": [{"as_of": s["as_of"], "mid": s["mid"]} for s in snapshots],
    }


def _parse_as_of(value):
    """Parse an ISO ``YYYY-MM-DD`` string. Raises `ValueError` on a bad format."""
    return dt.date.fromisoformat(value)


# ---------------------------------------------------------------------------
# health / catalog / prediction
# ---------------------------------------------------------------------------


@bp.get("/health")
def health():
    predictor = _predictor()
    return jsonify(
        {
            "status": "ok",
            "model_version": predictor.model_version,
            "n_catalog": len(predictor.catalog),
        }
    )


@bp.get("/sneakers")
def search_sneakers():
    q = request.args.get("q", "")
    if not q.strip():
        return _error("query parameter 'q' is required", 400)

    limit_raw = request.args.get("limit")
    limit = 10
    if limit_raw is not None:
        try:
            limit = int(limit_raw)
        except ValueError:
            return _error(f"limit must be an integer, got {limit_raw!r}", 400)

    results = _predictor().search(q, limit=limit)
    return jsonify(results)


@bp.get("/importance")
def importance():
    path = current_app.config["MODELS_DIR"] / "global_importance.json"
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return _error(f"global importance file not found at {path}", 404)
    records = [{"feature": name, "importance": value} for name, value in raw.items()]
    return jsonify(records)


@bp.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error("request body must be a JSON object", 400)

    sneaker_id = body.get("sneaker_id")
    size = body.get("size")
    if not sneaker_id or size is None:
        return _error("'sneaker_id' and 'size' are required", 400)

    as_of = None
    if body.get("as_of") is not None:
        try:
            as_of = _parse_as_of(body["as_of"])
        except ValueError:
            return _error(f"'as_of' must be an ISO date (YYYY-MM-DD), got {body['as_of']!r}", 400)

    try:
        result = _predictor().predict(sneaker_id, size, as_of=as_of)
    except KeyError as exc:
        return _error(str(exc), 404)
    except ValueError as exc:
        return _error(str(exc), 400)

    return jsonify(result)


# ---------------------------------------------------------------------------
# watchlist
# ---------------------------------------------------------------------------


@bp.get("/watchlist")
def list_watchlist():
    conn = db.get_db()
    predictor = _predictor()
    items = [_watchlist_item_response(conn, predictor, row) for row in db.list_items(conn)]
    return jsonify(items)


@bp.post("/watchlist")
def add_watchlist_item():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error("request body must be a JSON object", 400)

    sneaker_id = body.get("sneaker_id")
    size = body.get("size")
    if not sneaker_id or size is None:
        return _error("'sneaker_id' and 'size' are required", 400)

    predictor = _predictor()
    try:
        prediction = predictor.predict(sneaker_id, size)
    except KeyError as exc:
        return _error(str(exc), 404)
    except ValueError as exc:
        return _error(str(exc), 400)

    conn = db.get_db()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        item_id = db.add_item(conn, sneaker_id, prediction["size"], now)
    except sqlite3.IntegrityError:
        return _error(
            f"sneaker {sneaker_id!r} at size {prediction['size']} is already on the watchlist",
            409,
        )

    db.add_snapshot(
        conn,
        item_id,
        prediction["as_of"],
        prediction["low"],
        prediction["mid"],
        prediction["high"],
        prediction["extrapolated"],
        now,
    )

    item = db.get_item(conn, item_id)
    return jsonify(_watchlist_item_response(conn, predictor, item)), 201


@bp.delete("/watchlist/<int:item_id>")
def delete_watchlist_item(item_id: int):
    conn = db.get_db()
    if not db.delete_item(conn, item_id):
        return _error(f"no watchlist item with id {item_id}", 404)
    return "", 204


@bp.post("/watchlist/refresh")
def refresh_watchlist():
    body = request.get_json(silent=True) or {}
    advance_days = body.get("advance_days", 30)
    explicit_as_of = None
    if body.get("as_of") is not None:
        try:
            explicit_as_of = _parse_as_of(body["as_of"])
        except ValueError:
            return _error(f"'as_of' must be an ISO date (YYYY-MM-DD), got {body['as_of']!r}", 400)

    conn = db.get_db()
    predictor = _predictor()
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    refreshed = []
    for item in db.list_items(conn):
        snapshots = db.snapshots_for(conn, item["id"])
        if explicit_as_of is not None:
            new_as_of = explicit_as_of
        else:
            latest_as_of = dt.date.fromisoformat(snapshots[-1]["as_of"]) if snapshots else dt.date.today()
            new_as_of = latest_as_of + dt.timedelta(days=advance_days)

        prediction = predictor.predict(item["sneaker_id"], item["size"], as_of=new_as_of)
        db.add_snapshot(
            conn,
            item["id"],
            prediction["as_of"],
            prediction["low"],
            prediction["mid"],
            prediction["high"],
            prediction["extrapolated"],
            now,
        )
        refreshed.append(
            {
                "sneaker_id": item["sneaker_id"],
                "as_of": prediction["as_of"],
                "mid": prediction["mid"],
                "extrapolated": prediction["extrapolated"],
            }
        )

    return jsonify(refreshed)
