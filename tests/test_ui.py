"""The browser UI is served and wired to the API."""

from fastapi.testclient import TestClient

from clinicalgraph.api.main import app

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Clinical Knowledge Graph QA" in r.text


def test_static_js_served():
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "/ask" in r.text  # the UI calls the ask endpoint
