"""Tests for the Flask API (api/app.py, api/db.py, api/routes.py)."""

from __future__ import annotations

import json

import pytest

from sneakerml.api.app import create_app

BELUGA_SLUG = "adidas-yeezy-boost-350-low-v2-beluga"


@pytest.fixture
def app(tiny_models, tmp_path):
    """A fresh Flask app per test: isolated tmp_path DB, real tiny models."""
    application = create_app(
        {
            "TESTING": True,
            "DB_PATH": tmp_path / "test.db",
            "MODELS_DIR": tiny_models["models_dir"],
            "CATALOG_PATH": tiny_models["catalog_path"],
        }
    )
    yield application


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# health / search / predict
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert isinstance(body["model_version"], str)
    assert body["n_catalog"] > 0


def test_search_returns_beluga(client):
    resp = client.get("/api/sneakers?q=beluga")
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body, list)
    assert any(item["id"] == BELUGA_SLUG for item in body)
    first = body[0]
    for key in ("id", "sku", "display_name", "brand", "silhouette", "retail_price", "sizes_seen"):
        assert key in first


def test_search_requires_q(client):
    resp = client.get("/api/sneakers")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_happy_path(client):
    resp = client.post("/api/predict", json={"sneaker_id": BELUGA_SLUG, "size": 10.0})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["low"] <= body["mid"] <= body["high"]
    assert body["sneaker"]["id"] == BELUGA_SLUG
    assert isinstance(body["contributions"], list)
    assert body["contributions"]


def test_predict_unknown_id_404(client):
    resp = client.post("/api/predict", json={"sneaker_id": "no-such-shoe", "size": 10.0})
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_predict_bad_size_400(client):
    resp = client.post("/api/predict", json={"sneaker_id": BELUGA_SLUG, "size": 99.0})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_missing_fields_400(client):
    resp = client.post("/api/predict", json={"sneaker_id": BELUGA_SLUG})
    assert resp.status_code == 400


def test_predict_bad_as_of_400(client):
    resp = client.post(
        "/api/predict", json={"sneaker_id": BELUGA_SLUG, "size": 10.0, "as_of": "not-a-date"}
    )
    assert resp.status_code == 400


def test_importance(client, tiny_models):
    # tiny_models (conftest.py) trains models but doesn't write
    # global_importance.json -- that's a `sneakerml.cli train` side effect --
    # so write a small stand-in here to exercise the endpoint's shape.
    importance_path = tiny_models["models_dir"] / "global_importance.json"
    importance_path.write_text(json.dumps({"silhouette": 0.28, "colorway": 0.14}))

    resp = client.get("/api/importance")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == [
        {"feature": "silhouette", "importance": 0.28},
        {"feature": "colorway", "importance": 0.14},
    ]


def test_importance_missing_file_404(client, tiny_models):
    importance_path = tiny_models["models_dir"] / "global_importance.json"
    importance_path.unlink(missing_ok=True)

    resp = client.get("/api/importance")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# watchlist
# ---------------------------------------------------------------------------


def test_watchlist_add_list_refresh_delete(client):
    # empty at first
    resp = client.get("/api/watchlist")
    assert resp.status_code == 200
    assert resp.get_json() == []

    # add
    resp = client.post("/api/watchlist", json={"sneaker_id": BELUGA_SLUG, "size": 10.0})
    assert resp.status_code == 201
    created = resp.get_json()
    assert created["sneaker_id"] == BELUGA_SLUG
    assert created["size"] == 10.0
    assert created["display_name"]
    assert created["latest"] is not None
    assert len(created["history"]) == 1
    item_id = created["id"]

    # list shows 1 with a snapshot
    resp = client.get("/api/watchlist")
    assert resp.status_code == 200
    items = resp.get_json()
    assert len(items) == 1
    assert items[0]["id"] == item_id
    assert len(items[0]["history"]) == 1

    # duplicate add is 409
    resp = client.post("/api/watchlist", json={"sneaker_id": BELUGA_SLUG, "size": 10.0})
    assert resp.status_code == 409
    assert "error" in resp.get_json()

    # refresh adds a history point
    resp = client.post("/api/watchlist/refresh", json={"advance_days": 30})
    assert resp.status_code == 200
    refreshed = resp.get_json()
    assert len(refreshed) == 1
    assert refreshed[0]["sneaker_id"] == BELUGA_SLUG
    assert "extrapolated" in refreshed[0]

    resp = client.get("/api/watchlist")
    items = resp.get_json()
    assert len(items[0]["history"]) == 2
    assert items[0]["latest"]["as_of"] == items[0]["history"][-1]["as_of"]

    # delete
    resp = client.delete(f"/api/watchlist/{item_id}")
    assert resp.status_code == 204
    assert resp.data == b""

    resp = client.get("/api/watchlist")
    assert resp.get_json() == []


def test_watchlist_add_unknown_id_404(client):
    resp = client.post("/api/watchlist", json={"sneaker_id": "no-such-shoe", "size": 10.0})
    assert resp.status_code == 404


def test_watchlist_add_bad_size_400(client):
    resp = client.post("/api/watchlist", json={"sneaker_id": BELUGA_SLUG, "size": 99.0})
    assert resp.status_code == 400


def test_watchlist_delete_unknown_404(client):
    resp = client.delete("/api/watchlist/999")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_watchlist_delete_cascades_snapshots(client, app):
    resp = client.post("/api/watchlist", json={"sneaker_id": BELUGA_SLUG, "size": 10.0})
    item_id = resp.get_json()["id"]

    with app.app_context():
        from sneakerml.api import db

        conn = db.get_db()
        assert len(db.snapshots_for(conn, item_id)) == 1

    client.delete(f"/api/watchlist/{item_id}")

    with app.app_context():
        from sneakerml.api import db

        conn = db.get_db()
        # cascade actually removed the snapshot row, not just the watchlist row
        rows = conn.execute("SELECT * FROM snapshots WHERE watchlist_id = ?", (item_id,)).fetchall()
        assert rows == []


def test_watchlist_refresh_explicit_as_of_overrides(client):
    client.post("/api/watchlist", json={"sneaker_id": BELUGA_SLUG, "size": 10.0})
    resp = client.post("/api/watchlist/refresh", json={"as_of": "2020-01-01"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body[0]["as_of"] == "2020-01-01"
