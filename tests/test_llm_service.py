from types import SimpleNamespace

import pytest

from src.knowledge_base import POR_ID
from src.llm_service import LLMService, viola_guardrails
from src.mensagens import AVISO, CONDUTA
from src.rules import MotorDeRegras
from src.schemas import NivelRisco


def _triagem(marcador_id, valor):
    m = POR_ID[marcador_id]
    return MotorDeRegras.avaliar_triagem_completa([{"marcador": m, "valor": valor, "ref": m.ref, "ref_fonte": "base"}])


class _FakeClient:
    def __init__(self, resposta=None, erro=None):
        self.prompts = []

        def create(**kwargs):
            self.prompts.append(kwargs["messages"])
            if erro:
                raise erro
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=resposta))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _servico(client):
    s = LLMService()
    s.client = client
    return s


@pytest.mark.parametrize("texto", [
    "Você tem anemia.",
    "Você está com uma infecção.",
    "O diagnóstico é diabetes.",
    "Tome 500 mg de ferro por dia.",
    "Recomendo usar um suplemento de ferro.",
])
def test_guardrails_bloqueiam_diagnostico_e_prescricao(texto):
    assert viola_guardrails(texto)


def test_guardrails_permitem_texto_explicativo():
    assert not viola_guardrails("Sua hemoglobina está abaixo da referência, o que pode indicar anemia.")


def test_sem_chave_usa_texto_padrao():
    mensagem, fonte = _servico(None).gerar_mensagem_humanizada(_triagem("glicose", 92), "Ana Souza")
    assert fonte == "modelo"
    assert mensagem.startswith("Olá, Ana!")
    assert mensagem.endswith(AVISO)


def test_llm_gera_apenas_o_corpo_e_conduta_critica_e_fixa():
    client = _FakeClient(resposta="Sua hemoglobina está bem abaixo do esperado e a equipe já foi avisada.")
    mensagem, fonte = _servico(client).gerar_mensagem_humanizada(_triagem("hemoglobina", 6.0), "José Almeida")
    assert fonte == "llm"
    assert "Sua hemoglobina está bem abaixo" in mensagem
    assert CONDUTA[NivelRisco.CRITICO] in mensagem
    assert "192" in mensagem


def test_nome_do_paciente_nao_e_enviado_ao_llm():
    client = _FakeClient(resposta="Texto.")
    _servico(client).gerar_mensagem_humanizada(_triagem("glicose", 92), "Ana Souza")
    enviado = str(client.prompts)
    assert "Ana" not in enviado and "Souza" not in enviado


def test_resposta_que_viola_guardrail_e_descartada():
    client = _FakeClient(resposta="Você tem anemia grave, tome 300 mg de ferro.")
    mensagem, fonte = _servico(client).gerar_mensagem_humanizada(_triagem("hemoglobina", 6.0))
    assert fonte == "modelo"
    assert "anemia grave" not in mensagem


def test_erro_do_llm_usa_texto_padrao():
    client = _FakeClient(erro=TimeoutError("timeout"))
    mensagem, fonte = _servico(client).gerar_mensagem_humanizada(_triagem("glicose", 118))
    assert fonte == "modelo"
    assert CONDUTA[NivelRisco.ATENCAO] in mensagem
