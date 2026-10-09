import pytest

from src.knowledge_base import POR_ID
from src.parser import LabParser
from src.rules import MotorDeRegras
from src.schemas import NivelRisco


def _item(marcador_id, valor, ref=None, fonte="base"):
    m = POR_ID[marcador_id]
    return {"marcador": m, "valor": valor, "ref": ref or m.ref, "ref_fonte": fonte}


def _triagem(texto):
    return MotorDeRegras.avaliar_triagem_completa(LabParser.parse_texto(texto))


def test_hemoglobina_critica():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("hemoglobina", 6.2)])
    assert resultado.classificacao_geral == NivelRisco.CRITICO
    assert resultado.cor_badge == "#EF4444"
    assert resultado.exames_extraidos[0].direcao == "baixo"


def test_potassio_critico_alto():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("potassio", 6.6)])
    assert resultado.classificacao_geral == NivelRisco.CRITICO
    assert resultado.exames_extraidos[0].direcao == "alto"


def test_valor_fora_da_referencia_e_atencao():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("glicose", 118)])
    assert resultado.classificacao_geral == NivelRisco.ATENCAO
    assert resultado.cor_badge == "#F59E0B"


def test_referencia_do_laudo_tem_prioridade_sobre_a_base():
    # 16 g/dL é normal na base (12 a 17,5), mas acima da referência do laudo (12 a 15,5).
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("hemoglobina", 16, ref=(12, 15.5), fonte="laudo")])
    assert resultado.classificacao_geral == NivelRisco.ATENCAO


def test_limite_critico_vem_sempre_da_base():
    # Mesmo que o laudo traga uma referência larga, 6,0 g/dL continua crítico.
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("hemoglobina", 6.0, ref=(5, 15), fonte="laudo")])
    assert resultado.classificacao_geral == NivelRisco.CRITICO


def test_referencia_so_com_maximo():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("colesterol", 150)])
    assert resultado.classificacao_geral == NivelRisco.NORMAL


def test_risco_geral_e_o_pior_item_independente_da_ordem():
    itens = [_item("hemoglobina", 6.0), _item("glicose", 118), _item("plaquetas", 250000)]
    assert MotorDeRegras.avaliar_triagem_completa(itens).classificacao_geral == NivelRisco.CRITICO
    assert MotorDeRegras.avaliar_triagem_completa(itens[::-1]).classificacao_geral == NivelRisco.CRITICO


def test_fontes_e_guardrails_preenchidos():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("glicose", 118)])
    assert resultado.fontes == ["Glicose (jejum): quantidade de açúcar no sangue. Referência (adulto): 70 a 99 mg/dL."]
    assert any("retido para validação" in g for g in resultado.guardrails)


# Laudos de exemplo e da fila inicial do site (app/lib/triage.js e Simulator.js):
# a API precisa chegar ao mesmo nível de risco que o motor do navegador.
EXEMPLOS_DO_SITE = [
    ("""HEMOGRAMA COMPLETO
Hemoglobina: 14,2 g/dL      (VR: 12,0 a 17,5)
Hematócrito: 42 %           (VR: 36 a 52)
Leucócitos: 6.800 /mm³      (VR: 4.000 a 11.000)
Plaquetas: 245.000 /mm³     (VR: 150.000 a 450.000)

BIOQUÍMICA
Glicose: 88 mg/dL           (VR: 70 a 99)
Creatinina: 0,9 mg/dL       (VR: 0,6 a 1,3)""", NivelRisco.NORMAL, "Hemograma"),
    ("""HEMOGRAMA COMPLETO
Hemoglobina: 6,4 g/dL       (VR: 12,0 a 17,5)
Hematócrito: 20 %           (VR: 36 a 52)
Leucócitos: 5.200 /mm³      (VR: 4.000 a 11.000)
Plaquetas: 98.000 /mm³      (VR: 150.000 a 450.000)

ELETRÓLITOS
Potássio: 4,2 mEq/L         (VR: 3,5 a 5,1)""", NivelRisco.CRITICO, "Hemograma"),
    ("""PERFIL METABÓLICO
Glicose: 118 mg/dL          (VR: 70 a 99)
Colesterol total: 232 mg/dL (VR: < 190)
Triglicerídeos: 210 mg/dL   (VR: < 150)
Creatinina: 1,0 mg/dL       (VR: 0,6 a 1,3)
TSH: 2,1 µUI/mL             (VR: 0,4 a 4,5)""", NivelRisco.ATENCAO, "Perfil metabólico"),
    ("HEMOGRAMA\nHemoglobina: 11,1 g/dL\nLeucócitos: 12.400 /mm³\nPlaquetas: 210.000 /mm³", NivelRisco.ATENCAO, "Hemograma"),
    ("ELETRÓLITOS\nPotássio: 6,6 mEq/L\nSódio: 138 mEq/L\nCreatinina: 2,1 mg/dL", NivelRisco.CRITICO, "Eletrólitos"),
    ("PERFIL LIPÍDICO\nColesterol total: 248 mg/dL\nTriglicerídeos: 190 mg/dL", NivelRisco.ATENCAO, "Perfil lipídico"),
    ("GLICEMIA\nGlicose: 92 mg/dL\nTSH: 1,8 µUI/mL", NivelRisco.NORMAL, "Glicemia"),
]


@pytest.mark.parametrize("texto, nivel, tipo", EXEMPLOS_DO_SITE)
def test_paridade_com_exemplos_do_site(texto, nivel, tipo):
    dados = LabParser.parse_texto(texto)
    tipo_exame = LabParser.detectar_tipo_exame(texto, [d["marcador"].id for d in dados])
    resultado = MotorDeRegras.avaliar_triagem_completa(dados, tipo_exame)
    assert resultado.classificacao_geral == nivel
    assert resultado.tipo_exame == tipo
