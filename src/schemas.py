from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class NivelRisco(str, Enum):
    NORMAL = "NORMAL"
    ATENCAO = "ATENCAO"
    CRITICO = "CRITICO"


class ItemExame(BaseModel):
    parametro: str = Field(..., description="Nome do marcador (ex: Hemoglobina)")
    valor: float = Field(..., description="Valor numérico encontrado")
    unidade: str = Field(..., description="Unidade de medida (ex: g/dL)")
    ref_min: float = Field(..., description="Valor de referência mínimo")
    ref_max: float = Field(..., description="Valor de referência máximo")
    status_item: NivelRisco


class ResultadoTriagem(BaseModel):
    paciente_id: Optional[str] = "P-000"
    tipo_exame: str
    classificacao_geral: NivelRisco
    cor_badge: str
    exames_extraidos: List[ItemExame]
    mensagem_humanizada: Optional[str] = None


class LaudoEntradaRequest(BaseModel):
    paciente_id: str = Field(..., min_length=1)
    texto_laudo: str = Field(..., min_length=1)
