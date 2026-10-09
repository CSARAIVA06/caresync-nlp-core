import logging

from groq import Groq

from src.config import settings
from src.schemas import NivelRisco, ResultadoTriagem

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é o assistente virtual de comunicação hospitalar do CareSync.
Sua única função é traduzir resultados de exames para uma linguagem empática, clara e simples para pacientes leigos.

REGRAS DE SEGURANÇA E ÉTICA (GUARDRAILS):
1. NUNCA emita diagnósticos médicos definitivos (ex: NUNCA diga "Você tem anemia" ou "Você está com infecção").
2. NUNCA prescreva medicamentos ou tratamentos.
3. Se o status for NORMAL: Seja reconfortante e informe que as taxas estão adequadas.
4. Se o status for ATENCAO: Sugira de forma amável o agendamento de uma consulta de retorno com o médico.
5. Se o status for CRITICO: NUNCA cause pânico, mas informe de forma clara que uma notificação prioritária foi enviada à equipe de enfermagem do hospital e oriente busca por atendimento rápido.
"""

# Mensagens determinísticas usadas quando o LLM não está configurado ou falha.
MENSAGENS_FALLBACK = {
    NivelRisco.NORMAL: "Seus resultados estão dentro dos valores de referência. Continue acompanhando sua saúde com o seu médico.",
    NivelRisco.ATENCAO: "Alguns valores do seu exame estão fora da referência. Recomendamos agendar uma consulta de retorno com o seu médico.",
    NivelRisco.CRITICO: "Alguns valores do seu exame precisam de avaliação rápida. Uma notificação prioritária foi enviada à equipe de enfermagem; procure atendimento o quanto antes.",
}


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    def gerar_mensagem_humanizada(self, triagem: ResultadoTriagem) -> str:
        fallback = MENSAGENS_FALLBACK[triagem.classificacao_geral]
        if not self.client:
            return fallback

        prompt_usuario = f"""
        Classificação do Exame: {triagem.classificacao_geral.value}
        Exames Avaliados: {[e.model_dump(mode="json") for e in triagem.exames_extraidos]}

        Gere uma mensagem curta em português direcionada ao paciente baseada exclusivamente nessas informações.
        """

        try:
            response = self.client.chat.completions.create(
                model=settings.DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_usuario},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            conteudo = response.choices[0].message.content
            return conteudo.strip() if conteudo else fallback
        except Exception:
            # A triagem é determinística; uma falha do LLM não deve derrubar a resposta.
            logger.exception("Falha ao gerar mensagem humanizada via LLM")
            return fallback
