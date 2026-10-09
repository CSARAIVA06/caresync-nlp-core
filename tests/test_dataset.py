import random

import pytest

from scripts.gerar_dataset import LAYOUTS_OOD, LAYOUTS_VISTOS, gerar_laudo, tokens_bio
from src.parser import LabParser
from src.schemas import NivelRisco


@pytest.mark.parametrize("layout", LAYOUTS_VISTOS + LAYOUTS_OOD)
def test_entidades_apontam_para_o_texto_certo(layout):
    rng = random.Random(0)
    for i, alvo in enumerate(NivelRisco):
        laudo = gerar_laudo(rng, i, layout, alvo)
        for e in laudo["entidades"]:
            assert laudo["texto"][e["inicio"]:e["fim"]] == e["texto"]
        assert {e["label"] for e in laudo["entidades"]} >= {"EXAME", "VALOR", "UNIDADE"}
        assert laudo["risco"] in {"NORMAL", "ATENCAO", "CRITICO"}


def test_gerador_e_deterministico():
    a = gerar_laudo(random.Random(7), 1, "tabela_pipe", NivelRisco.CRITICO)
    b = gerar_laudo(random.Random(7), 1, "tabela_pipe", NivelRisco.CRITICO)
    assert a == b


@pytest.mark.parametrize("layout", LAYOUTS_VISTOS)
def test_regex_le_os_formatos_vistos(layout):
    # Os formatos de treino são legíveis pelo regex (a linha de base é justa nesses formatos).
    rng = random.Random(1)
    laudo = gerar_laudo(rng, 0, layout, NivelRisco.ATENCAO)
    extraido = {(d["marcador"].id, d["valor"]) for d in LabParser.parse_texto(laudo["texto"])}
    assert extraido == {(i["id"], i["valor"]) for i in laudo["itens"]}


def test_tokens_bio():
    texto = "Hb: 13,5 g/dL"
    entidades = [{"inicio": 0, "fim": 2, "label": "EXAME"}, {"inicio": 4, "fim": 8, "label": "VALOR"},
                 {"inicio": 9, "fim": 13, "label": "UNIDADE"}]
    tokens, tags = tokens_bio(texto, entidades)
    assert list(zip(tokens, tags)) == [("Hb", "B-EXAME"), (":", "O"), ("13,5", "B-VALOR"),
                                       ("g", "B-UNIDADE"), ("/", "I-UNIDADE"), ("dL", "I-UNIDADE")]


def test_textos_alvo_nao_violam_guardrails():
    from src.llm_service import viola_guardrails
    from src.metricas import fidelidade_numerica

    rng = random.Random(3)
    for i in range(60):
        laudo = gerar_laudo(rng, i, LAYOUTS_VISTOS[i % len(LAYOUTS_VISTOS)], list(NivelRisco)[i % 3])
        assert not viola_guardrails(laudo["explicacao_alvo"])
        assert fidelidade_numerica(laudo["entrada_seq2seq"], laudo["explicacao_alvo"])
