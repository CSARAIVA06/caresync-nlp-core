from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.llm_service import LLMService
from src.parser import LabParser
from src.rules import MotorDeRegras
from src.schemas import LaudoEntradaRequest, ResultadoTriagem

app = FastAPI(title="CareSync NLP Core API", version="1.1.0")

# Permitir chamadas do site no Vercel / Next.js (restrinja CORS_ORIGINS em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def criar_extrator():
    """Regex (padrão) ou NER treinado; o modelo é carregado uma vez, na inicialização."""
    if settings.EXTRATOR == "ner":
        from src.ner import NerExtractor

        return NerExtractor(settings.NER_MODELO_DIR)
    return LabParser


extrator = criar_extrator()
llm_service = LLMService()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "CareSync NLP Engine",
        "extrator": settings.EXTRATOR,
        "gerador": llm_service.gerador,
    }


@app.post("/api/v1/triagem", response_model=ResultadoTriagem)
def processar_laudo(payload: LaudoEntradaRequest):
    # 1. Extração (regex ou NER)
    dados_brutos = extrator.parse_texto(payload.texto_laudo)
    if not dados_brutos:
        raise HTTPException(status_code=400, detail="Não foi possível extrair marcadores válidos do texto enviado.")
    tipo_exame = LabParser.detectar_tipo_exame(payload.texto_laudo, [d["marcador"].id for d in dados_brutos])

    # 2. Motor de Regras
    resultado = MotorDeRegras.avaliar_triagem_completa(dados_brutos, tipo_exame)
    resultado.paciente_id = payload.paciente_id

    # 3. Mensagem humanizada via LLM (com texto padrão como fallback)
    resultado.mensagem_humanizada, resultado.fonte_mensagem = llm_service.gerar_mensagem_humanizada(
        resultado, payload.paciente_nome
    )

    return resultado
