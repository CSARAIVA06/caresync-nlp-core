import pytest

from src.parser import LabParser, normalizar, parse_numero

LAUDO = """HEMOGRAMA COMPLETO
Hemoglobina: 14,2 g/dL      (VR: 12,0 a 15,5)
Hematócrito: 42 %           (VR: 36 a 52)
Leucócitos: 6.800 /mm³      (VR: 4.000 a 11.000)
Plaquetas: 245.000 /mm³     (VR: 150.000 a 450.000)
Colesterol total: 232 mg/dL (VR: < 190)
"""


def _por_id(texto):
    return {r["marcador"].id: r for r in LabParser.parse_texto(texto)}


@pytest.mark.parametrize("texto, resto, esperado", [
    ("13,5", "", 13.5),
    ("13.5", "", 13.5),
    ("250.000", "", 250000.0),
    ("1.250,5", "", 1250.5),
    ("250", " mil /mm³", 250000.0),
    ("85", "", 85.0),
])
def test_parse_numero(texto, resto, esperado):
    assert parse_numero(texto, resto) == esperado


def test_normalizar_remove_acentos():
    assert normalizar("Leucócitos POTÁSSIO") == "leucocitos potassio"


def test_extrai_valores_e_referencias_do_laudo():
    r = _por_id(LAUDO)
    assert set(r) == {"hemoglobina", "hematocrito", "leucocitos", "plaquetas", "colesterol"}
    assert r["hemoglobina"]["valor"] == 14.2
    assert r["hemoglobina"]["ref"] == (12.0, 15.5)
    assert r["hemoglobina"]["ref_fonte"] == "laudo"
    assert r["leucocitos"]["valor"] == 6800.0
    assert r["plaquetas"]["valor"] == 245000.0
    assert r["colesterol"]["ref"] == (None, 190.0)


def test_sem_referencia_no_laudo_usa_base():
    r = _por_id("Glicose: 92 mg/dL")
    assert r["glicose"]["ref"] == (70, 99)
    assert r["glicose"]["ref_fonte"] == "base"


def test_referencia_em_outra_escala_e_ignorada():
    # "4,0 a 11,0" (em mil) não pode ser comparado com 6800/mm³.
    r = _por_id("Leucócitos: 6,8 mil/mm³ (VR: 4,0 a 11,0)")
    assert r["leucocitos"]["valor"] == 6800.0
    assert r["leucocitos"]["ref_fonte"] == "base"


def test_formato_referencia_por_extenso():
    r = _por_id("Hemoglobina: 13,5 g/dL    Referência: 12,0 a 15,5")
    assert r["hemoglobina"]["valor"] == 13.5
    assert r["hemoglobina"]["ref"] == (12.0, 15.5)


def test_varios_marcadores_na_mesma_linha():
    r = _por_id("Sódio 138 mEq/L  Potássio 4,2 mEq/L")
    assert r["sodio"]["valor"] == 138
    assert r["potassio"]["valor"] == 4.2


def test_hemoglobina_glicada_nao_e_hemoglobina():
    assert _por_id("Hemoglobina glicada: 6,1 %") == {}


def test_colesterol_hdl_nao_e_colesterol_total():
    r = _por_id("Colesterol HDL: 45 mg/dL\nColesterol total: 180 mg/dL")
    assert r["colesterol"]["valor"] == 180


def test_data_nao_e_confundida_com_valor():
    r = _por_id("Glicose (coleta 12/03/2026): 92 mg/dL")
    assert r["glicose"]["valor"] == 92


def test_texto_sem_marcadores_retorna_lista_vazia():
    assert LabParser.parse_texto("Paciente compareceu para coleta.") == []


@pytest.mark.parametrize("texto, ids, esperado", [
    ("HEMOGRAMA COMPLETO\n...", ["hemoglobina"], "Hemograma"),
    ("PERFIL LIPÍDICO", ["colesterol"], "Perfil lipídico"),
    ("sem título", ["hemoglobina", "plaquetas"], "Hemograma"),
    ("sem título", ["glicose", "tsh"], "Exames laboratoriais"),
])
def test_detectar_tipo_exame(texto, ids, esperado):
    assert LabParser.detectar_tipo_exame(texto, ids) == esperado
