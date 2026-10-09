import pytest

from src.metricas import (
    comparar_extracoes,
    fidelidade_numerica,
    flesch_ptbr,
    prf,
    relatorio_classificacao,
    rouge_l,
)


def test_prf():
    m = prf(tp=8, fp=2, fn=2)
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.8)


def test_valor_errado_conta_como_fp_e_fn():
    assert comparar_extracoes([("hemoglobina", 13.5)], [("hemoglobina", 12.0)]) == (0, 1, 1)
    assert comparar_extracoes([("hemoglobina", 13.5)], [("hemoglobina", 13.5)]) == (1, 0, 0)


def test_relatorio_conta_criticos_perdidos():
    r = relatorio_classificacao(["CRITICO", "CRITICO", "NORMAL"], ["CRITICO", "ATENCAO", "NORMAL"])
    assert r["criticos_perdidos"] == 1
    assert r["por_classe"]["CRITICO"]["recall"] == pytest.approx(0.5)
    assert r["matriz_confusao"] == [[1, 0, 0], [0, 0, 0], [0, 1, 1]]


def test_rouge_l_considera_acentos():
    assert rouge_l("a glicose está alta", "a glicose está alta") == pytest.approx(1.0)
    assert rouge_l("potássio está alto", "sódio está normal") == pytest.approx(1 / 3)
    assert rouge_l("texto", "") == 0.0


def test_flesch_texto_simples_e_mais_facil_que_tecnico():
    simples = "Seu exame está bom. A taxa de açúcar está normal. Não é preciso fazer nada agora."
    tecnico = ("Hemoglobina glicosilada com concentração plasmática eritrocitária significativamente "
               "aumentada, compatível com hiperglicemia persistente e descompensação metabólica crônica.")
    assert flesch_ptbr(simples) > flesch_ptbr(tecnico)


def test_fidelidade_numerica():
    entrada = "triagem: CRITICO | Hemoglobina: 6,4 g/dL (ref 12 a 17,5 g/dL; baixo)"
    assert fidelidade_numerica(entrada, "Sua hemoglobina está em 6,4 g/dL, abaixo de 12 a 17,5 g/dL.")
    assert not fidelidade_numerica(entrada, "Sua hemoglobina está em 5,9 g/dL.")


@pytest.mark.parametrize("palavra, silabas", [
    ("casa", 2), ("pão", 1), ("que", 1), ("água", 2), ("leite", 2),
    ("poema", 3), ("teatro", 3), ("área", 3), ("coelho", 3), ("saída", 3), ("proteína", 4),
])
def test_contagem_de_silabas(palavra, silabas):
    from src.metricas import _silabas
    assert _silabas(palavra) == silabas


@pytest.mark.parametrize("gerado", [
    "Suas plaquetas estão em 245 mil /mm³.",
    "Suas plaquetas estão em 245000 /mm³.",
    "A referência começa em 150.000 e a hemoglobina está em 12,0 g/dL.",
])
def test_fidelidade_aceita_outra_grafia_do_mesmo_numero(gerado):
    entrada = "Plaquetas: 245.000 /mm³ (ref 150.000 a 450.000 /mm³) | Hemoglobina: 12 g/dL"
    assert fidelidade_numerica(entrada, gerado)
