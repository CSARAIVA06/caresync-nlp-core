from typing import Any, Dict, List

from src.schemas import ItemExame, NivelRisco, ResultadoTriagem

# Valores de pânico: abaixo destes limites o item é CRITICO independentemente da referência.
LIMITES_CRITICOS_MIN = {
    "hemoglobina": 7.0,
    "plaquetas": 50000,
}

MAPA_CORES = {
    NivelRisco.NORMAL: "#22C55E",
    NivelRisco.ATENCAO: "#F59E0B",
    NivelRisco.CRITICO: "#EF4444",
}

_PESO = {NivelRisco.NORMAL: 0, NivelRisco.ATENCAO: 1, NivelRisco.CRITICO: 2}


class MotorDeRegras:
    """Aplica lógica determinística de estratificação de risco clínico."""

    @staticmethod
    def classificar_item(item: Dict[str, Any]) -> ItemExame:
        val = item["valor"]
        rmin = item["ref_min"]
        rmax = item["ref_max"]
        limite_critico = LIMITES_CRITICOS_MIN.get(item["parametro"].lower())

        if limite_critico is not None and val < limite_critico:
            status = NivelRisco.CRITICO
        elif val < rmin or val > rmax:
            status = NivelRisco.ATENCAO
        else:
            status = NivelRisco.NORMAL

        return ItemExame(
            parametro=item["parametro"],
            valor=val,
            unidade=item["unidade"],
            ref_min=rmin,
            ref_max=rmax,
            status_item=status,
        )

    @classmethod
    def avaliar_triagem_completa(cls, exames_parseados: List[Dict[str, Any]]) -> ResultadoTriagem:
        itens = [cls.classificar_item(item) for item in exames_parseados]
        # O risco geral é o pior risco entre os itens.
        risco_geral = max((i.status_item for i in itens), key=_PESO.get, default=NivelRisco.NORMAL)

        return ResultadoTriagem(
            tipo_exame="Análise de Sangue / Hemograma",
            classificacao_geral=risco_geral,
            cor_badge=MAPA_CORES[risco_geral],
            exames_extraidos=itens,
        )
