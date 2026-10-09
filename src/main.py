from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.llm_service import LLMService
from src.parser import LabParser
from src.rules import MotorDeRegras
from src.schemas import LaudoEntradaRequest, ResultadoTriagem

app = FastAPI(title="CareSync NLP Core API", version="1.0.0")

# Permitir chamadas do Vercel / Next.js (restrinja CORS_ORIGINS em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

llm_service = LLMService()


@app.get("/")
def health_check():
    return {"status": "ok", "service": "CareSync NLP Engine"}


@app.post("/api/v1/triagem", response_model=ResultadoTriagem)
def processar_laudo(payload: LaudoEntradaRequest):
    # 1. Parsing
    dados_brutos = LabParser.parse_texto(payload.texto_laudo)
    if not dados_brutos:
        raise HTTPException(status_code=400, detail="Não foi possível extrair marcadores válidos do texto enviado.")

    # 2. Motor de Regras
    resultado = MotorDeRegras.avaliar_triagem_completa(dados_brutos)
    resultado.paciente_id = payload.paciente_id

    # 3. Geração de Mensagem Humanizada via LLM (com fallback determinístico)
    resultado.mensagem_humanizada = llm_service.gerar_mensagem_humanizada(resultado)

    return resultado
