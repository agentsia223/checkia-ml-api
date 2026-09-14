from tests.conftest import FAKE_ASR, FAKE_MT


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_models_reflects_registry(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["translation"] == FAKE_MT
    assert body["asr"] == FAKE_ASR
    assert body["asr_quantization"] == "int8"
