from typing import Any, Dict, List, Optional, Tuple

from src.knowledge_base import Marcador, formatar_faixa
from src.schemas import ItemExame, NivelRisco, ResultadoTriagem

MAPA_CORES = {
    NivelRisco.NORMAL: "#22C55E",
    NivelRisco.ATENCAO: "#F59E0B",
    NivelRisco.CRITICO: "#EF4444",
}

_PESO = {NivelRisco.NORMAL: 0, NivelRisco.ATENCAO: 1, NivelRisco.CRITICO: 2}


class MotorDeRegras:
    """Aplica lógica determinística de estratificação de risco clínico."""

    @staticmethod
    def classificar_valor(marcador: Marcador, valor: float, ref) -> Tuple[NivelRisco, Optional[str]]:
        ref_min, ref_max = ref
        crit_min, crit_max = marcador.critico

        if ref_min is not None and valor < ref_min:
            direcao = "baixo"
        elif ref_max is not None and valor > ref_max:
            direcao = "alto"
        else:
            direcao = None

        # Valores de pânico vêm sempre da base de conhecimento, nunca do laudo.
        if (crit_min is not None and valor < crit_min) or (crit_max is not None and valor > crit_max):
            return NivelRisco.CRITICO, direcao
        if direcao:
            return NivelRisco.ATENCAO, direcao
        return NivelRisco.NORMAL, None

    @classmethod
    def classificar_item(cls, item: Dict[str, Any]) -> ItemExame:
        marcador: Marcador = item["marcador"]
        status, direcao = cls.classificar_valor(marcador, item["valor"], item["ref"])
        return ItemExame(
            id=marcador.id,
            parametro=marcador.nome,
            valor=item["valor"],
            unidade=marcador.unidade,
            ref_min=item["ref"][0],
            ref_max=item["ref"][1],
            ref_fonte=item["ref_fonte"],
            status_item=status,
            direcao=direcao,
        )

    @classmethod
    def avaliar_triagem_completa(
        cls, exames_parseados: List[Dict[str, Any]], tipo_exame: str = "Exames laboratoriais"
    ) -> ResultadoTriagem:
        itens = [cls.classificar_item(item) for item in exames_parseados]
        # O risco geral é o pior risco entre os itens.
        risco_geral = max((i.status_item for i in itens), key=_PESO.get, default=NivelRisco.NORMAL)

        return ResultadoTriagem(
            tipo_exame=tipo_exame,
            classificacao_geral=risco_geral,
            cor_badge=MAPA_CORES[risco_geral],
            exames_extraidos=itens,
            fontes=[
                f"{e['marcador'].nome}: {e['marcador'].explicacao}. "
                f"Referência (adulto): {formatar_faixa(e['marcador'].ref, e['marcador'].unidade)}."
                for e in exames_parseados
            ],
            guardrails=[
                "Nenhum diagnóstico é afirmado, apenas a explicação do resultado.",
                "Toda mensagem orienta a avaliação com um profissional de saúde.",
                "Classificação de risco feita por regras determinísticas, não pelo modelo.",
                "Risco baixo: liberado para envio automático."
                if risco_geral == NivelRisco.NORMAL
                else "Risco moderado ou crítico: retido para validação da enfermagem.",
            ],
        )
