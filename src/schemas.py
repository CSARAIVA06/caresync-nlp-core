from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class NivelRisco(str, Enum):
    NORMAL = "NORMAL"
    ATENCAO = "ATENCAO"
    CRITICO = "CRITICO"


class ItemExame(BaseModel):
    id: str = Field(..., description="Identificador do marcador na base de conhecimento (ex: hemoglobina)")
    parametro: str = Field(..., description="Nome do marcador (ex: Hemoglobina)")
    valor: float = Field(..., description="Valor numérico encontrado")
    unidade: str = Field(..., description="Unidade de medida (ex: g/dL)")
    ref_min: Optional[float] = Field(None, description="Valor de referência mínimo (None = sem limite inferior)")
    ref_max: Optional[float] = Field(None, description="Valor de referência máximo (None = sem limite superior)")
    ref_fonte: Literal["laudo", "base"] = Field(..., description="Se a referência veio do próprio laudo ou da base de conhecimento")
    status_item: NivelRisco
    direcao: Optional[Literal["baixo", "alto"]] = Field(None, description="Lado em que o valor saiu da referência")


class ResultadoTriagem(BaseModel):
    paciente_id: Optional[str] = "P-000"
    tipo_exame: str
    classificacao_geral: NivelRisco
    cor_badge: str
    exames_extraidos: List[ItemExame]
    mensagem_humanizada: Optional[str] = None
    fonte_mensagem: Optional[Literal["llm", "ptt5", "modelo"]] = Field(
        None, description="'llm' (Groq), 'ptt5' (modelo local treinado) ou 'modelo' (texto padrão)"
    )
    fontes: List[str] = Field(default_factory=list, description="Trechos da base de conhecimento usados (RAG)")
    guardrails: List[str] = Field(default_factory=list, description="Regras de segurança aplicadas")


class LaudoEntradaRequest(BaseModel):
    paciente_id: str = Field("P-000", max_length=64)
    # Usado só na saudação; nunca é enviado ao LLM.
    paciente_nome: Optional[str] = Field(None, max_length=120)
    texto_laudo: str = Field(..., min_length=1, max_length=10_000)
