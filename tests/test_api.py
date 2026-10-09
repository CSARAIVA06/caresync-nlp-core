from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_triagem_critica():
    resp = client.post("/api/v1/triagem", json={
        "paciente_id": "P-123",
        "texto_laudo": "Hemoglobina: 6,2 g/dL Referência: 12,0 a 15,5",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["paciente_id"] == "P-123"
    assert body["classificacao_geral"] == "CRITICO"
    assert body["mensagem_humanizada"]


def test_laudo_sem_marcadores_retorna_400():
    resp = client.post("/api/v1/triagem", json={"paciente_id": "P-1", "texto_laudo": "texto qualquer"})
    assert resp.status_code == 400
