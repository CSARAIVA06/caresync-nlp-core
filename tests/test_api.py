from fastapi.testclient import TestClient

from src.main import app, llm_service

client = TestClient(app)
# Os testes da API não chamam o LLM de verdade.
llm_service.client = None


def test_health_check():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["extrator"] == "regex"
    assert isinstance(body["llm"], bool)


def test_triagem_critica():
    resp = client.post("/api/v1/triagem", json={
        "paciente_id": "P-123",
        "paciente_nome": "José Almeida",
        "texto_laudo": "HEMOGRAMA\nHemoglobina: 6,4 g/dL (VR: 12,0 a 17,5)\nPlaquetas: 98.000 /mm³",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["paciente_id"] == "P-123"
    assert body["classificacao_geral"] == "CRITICO"
    assert body["tipo_exame"] == "Hemograma"
    assert body["fonte_mensagem"] == "modelo"
    assert body["mensagem_humanizada"].startswith("Olá, José!")
    hb = body["exames_extraidos"][0]
    assert hb["id"] == "hemoglobina"
    assert hb["ref_fonte"] == "laudo"
    assert hb["direcao"] == "baixo"


def test_paciente_e_opcional():
    resp = client.post("/api/v1/triagem", json={"texto_laudo": "Glicose: 92 mg/dL"})
    assert resp.status_code == 200
    assert resp.json()["classificacao_geral"] == "NORMAL"


def test_laudo_sem_marcadores_retorna_400():
    resp = client.post("/api/v1/triagem", json={"texto_laudo": "texto qualquer"})
    assert resp.status_code == 400


def test_laudo_muito_grande_e_rejeitado():
    resp = client.post("/api/v1/triagem", json={"texto_laudo": "Glicose: 92\n" * 2000})
    assert resp.status_code == 422


def test_cors_permite_o_site():
    resp = client.options("/api/v1/triagem", headers={
        "Origin": "https://caresync-eta-ten.vercel.app",
        "Access-Control-Request-Method": "POST",
    })
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] in ("*", "https://caresync-eta-ten.vercel.app")
