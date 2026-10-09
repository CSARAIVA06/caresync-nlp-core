from src.rules import MotorDeRegras
from src.schemas import NivelRisco


def _item(parametro, valor, ref_min, ref_max, unidade=""):
    return {"parametro": parametro, "valor": valor, "unidade": unidade, "ref_min": ref_min, "ref_max": ref_max}


def test_hemoglobina_critica():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("Hemoglobina", 6.2, 12.0, 15.5, "g/dL")])
    assert resultado.classificacao_geral == NivelRisco.CRITICO
    assert resultado.cor_badge == "#EF4444"


def test_plaquetas_criticas():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("Plaquetas", 30000, 150000, 450000)])
    assert resultado.classificacao_geral == NivelRisco.CRITICO


def test_valor_fora_da_referencia_e_atencao():
    resultado = MotorDeRegras.avaliar_triagem_completa([_item("Ferritina", 8, 15, 150, "ng/mL")])
    assert resultado.classificacao_geral == NivelRisco.ATENCAO
    assert resultado.cor_badge == "#F59E0B"


def test_valores_normais():
    resultado = MotorDeRegras.avaliar_triagem_completa([
        _item("Hemoglobina", 13.5, 12.0, 15.5, "g/dL"),
        _item("Plaquetas", 250000, 150000, 450000),
    ])
    assert resultado.classificacao_geral == NivelRisco.NORMAL
    assert resultado.cor_badge == "#22C55E"


def test_risco_geral_e_o_pior_item_independente_da_ordem():
    itens = [
        _item("Hemoglobina", 6.0, 12.0, 15.5),
        _item("Ferritina", 8, 15, 150),
        _item("Plaquetas", 250000, 150000, 450000),
    ]
    assert MotorDeRegras.avaliar_triagem_completa(itens).classificacao_geral == NivelRisco.CRITICO
    assert MotorDeRegras.avaliar_triagem_completa(itens[::-1]).classificacao_geral == NivelRisco.CRITICO
