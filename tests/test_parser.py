import pytest

from src.parser import LabParser, parse_numero

LAUDO = """
HEMOGRAMA COMPLETO
Hemoglobina: 13,5 g/dL    Referência: 12,0 a 15,5
Leucócitos: 7.500 /mm³    Referência: 4.000 a 11.000
Plaquetas: 250.000 /mm³   Referência: 150.000 a 450.000
Ferritina: 85 ng/mL       Referência: 15 a 150
"""


@pytest.mark.parametrize("texto, esperado", [
    ("13,5", 13.5),
    ("13.5", 13.5),
    ("250.000", 250000.0),
    ("1.250,5", 1250.5),
    ("85", 85.0),
])
def test_parse_numero(texto, esperado):
    assert parse_numero(texto) == esperado


def test_extrai_todos_os_marcadores():
    resultados = {r["parametro"]: r for r in LabParser.parse_texto(LAUDO)}

    assert set(resultados) == {"Hemoglobina", "Leucocitos", "Plaquetas", "Ferritina"}
    assert resultados["Hemoglobina"] == {
        "parametro": "Hemoglobina", "valor": 13.5, "unidade": "g/dL", "ref_min": 12.0, "ref_max": 15.5,
    }
    assert resultados["Leucocitos"]["valor"] == 7500.0
    assert resultados["Plaquetas"]["valor"] == 250000.0
    assert resultados["Plaquetas"]["ref_min"] == 150000.0
    assert resultados["Ferritina"]["unidade"] == "ng/mL"


def test_texto_sem_marcadores_retorna_lista_vazia():
    assert LabParser.parse_texto("Paciente compareceu para coleta.") == []


@pytest.mark.parametrize("primeira_linha", ["Hemoglobina: 13,5 g/dL", "Hemoglobina: 13,5"])
def test_referencia_em_outra_linha_nao_e_capturada(primeira_linha):
    # Evita casar o valor de um marcador com a referência de outro.
    texto = f"{primeira_linha}\nFerritina: 85 ng/mL Referência: 15 a 150"
    parametros = [r["parametro"] for r in LabParser.parse_texto(texto)]
    assert parametros == ["Ferritina"]
