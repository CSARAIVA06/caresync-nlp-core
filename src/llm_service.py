import logging
import re
from typing import Optional, Tuple

from groq import Groq

from src.config import settings
from src.knowledge_base import POR_ID, formatar_numero
from src.mensagens import corpo_padrao, faixa_item, montar_mensagem
from src.metricas import fidelidade_numerica
from src.schemas import ResultadoTriagem

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é o assistente virtual de comunicação hospitalar do CareSync.
Sua única função é explicar resultados de exames em linguagem empática, clara e simples para pacientes leigos.

REGRAS DE SEGURANÇA E ÉTICA (GUARDRAILS):
1. NUNCA emita diagnósticos médicos definitivos (ex: NUNCA diga "Você tem anemia" ou "Você está com infecção"). Use "pode indicar", "pode estar relacionado".
2. NUNCA prescreva medicamentos, doses, dietas ou tratamentos.
3. Use SOMENTE os valores, referências e explicações fornecidos. Não invente números nem outros exames.
4. Se o status for NORMAL: seja reconfortante e informe que as taxas estão adequadas.
5. Se o status for ATENCAO: explique com calma o que está fora da referência, sem alarmar.
6. Se o status for CRITICO: NUNCA cause pânico, mas deixe claro que alguns valores precisam de avaliação rápida e que a equipe de saúde do hospital já foi avisada.

FORMATO:
- Escreva apenas a explicação dos resultados, em português do Brasil, com no máximo 120 palavras.
- NÃO escreva saudação, despedida, assinatura, orientação de conduta (agendar consulta, ir ao pronto-socorro) nem aviso legal: essas partes são adicionadas pelo sistema.
"""

# Verificação pós-geração: se o texto violar um guardrail, usamos o texto padrão.
PADROES_PROIBIDOS = [
    re.compile(r"\bvoc[eê]\s+(tem|est[aá]\s+com|possui|sofre\s+de)\b", re.IGNORECASE),
    re.compile(r"\b(diagn[oó]stico\s+(de|é)|foi\s+diagnosticad[oa])\b", re.IGNORECASE),
    re.compile(r"\b(tome|tomar|use|usar|ingira|ingerir)\b[^.]{0,40}\b(\d+\s*(mg|ml|g|comprimidos?|gotas)|rem[eé]dio|medicamento|suplemento)", re.IGNORECASE),
    re.compile(r"\bprescrev", re.IGNORECASE),
]


def viola_guardrails(texto: str) -> bool:
    return any(p.search(texto) for p in PADROES_PROIBIDOS)


def contexto_rag(triagem: ResultadoTriagem) -> str:
    linhas = []
    for i in triagem.exames_extraidos:
        m = POR_ID[i.id]
        situacao = {"baixo": f"abaixo da referência ({m.baixo})", "alto": f"acima da referência ({m.alto})"}.get(
            i.direcao, "dentro da referência"
        )
        linhas.append(
            f"- {i.parametro}: {formatar_numero(i.valor)} {i.unidade}; referência {faixa_item(i)}; "
            f"status {i.status_item.value}; {situacao}. O que é: {m.explicacao}."
        )
    return "\n".join(linhas)


class LLMService:
    def __init__(self):
        gerador = settings.GERADOR
        if gerador == "auto":
            gerador = "groq" if settings.GROQ_API_KEY else "padrao"
        self.gerador = gerador
        self.client = (
            Groq(api_key=settings.GROQ_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=1)
            if gerador == "groq" and settings.GROQ_API_KEY
            else None
        )
        # Modelo PTT5 treinado localmente (inferência offline, nenhum dado sai do servidor).
        self.local = None
        if gerador == "ptt5":
            from src.gerador_local import GeradorPTT5

            self.local = GeradorPTT5(settings.PTT5_MODELO_DIR)

    @staticmethod
    def _validar(corpo: Optional[str], contexto: str, origem: str) -> Optional[str]:
        """Guardrails pós-geração: sem diagnóstico/prescrição e sem números inventados."""
        corpo = (corpo or "").strip()
        if not corpo:
            return None
        if viola_guardrails(corpo):
            logger.warning("Texto do %s descartado por violar guardrails: %r", origem, corpo)
            return None
        if not fidelidade_numerica(contexto, corpo):
            logger.warning("Texto do %s descartado por citar números ausentes na entrada: %r", origem, corpo)
            return None
        return corpo

    def _gerar_corpo_ptt5(self, triagem: ResultadoTriagem) -> Optional[str]:
        from src.gerador_local import entrada_seq2seq

        return self._validar(self.local.gerar(triagem), entrada_seq2seq(triagem), "PTT5")

    def _gerar_corpo_llm(self, triagem: ResultadoTriagem) -> Optional[str]:
        # Dados do paciente (nome, id) nunca são enviados ao LLM: só os resultados.
        prompt_usuario = (
            f"Classificação geral do exame: {triagem.classificacao_geral.value}\n"
            f"Tipo de exame: {triagem.tipo_exame}\n"
            f"Resultados e base de conhecimento:\n{contexto_rag(triagem)}"
        )
        response = self.client.chat.completions.create(
            model=settings.DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return self._validar(response.choices[0].message.content, prompt_usuario, "LLM")

    def gerar_mensagem_humanizada(self, triagem: ResultadoTriagem, nome: Optional[str] = None) -> Tuple[str, str]:
        """Retorna (mensagem, fonte): fonte 'ptt5' (modelo local), 'llm' (Groq) ou 'modelo' (texto padrão)."""
        corpo, fonte = None, "modelo"
        try:
            if self.local is not None:
                corpo, fonte = self._gerar_corpo_ptt5(triagem), "ptt5"
            elif self.client is not None:
                corpo, fonte = self._gerar_corpo_llm(triagem), "llm"
        except Exception:
            # A triagem é determinística; uma falha do gerador não deve derrubar a resposta.
            logger.exception("Falha ao gerar mensagem humanizada")
        if not corpo:
            fonte = "modelo"
        return montar_mensagem(triagem, corpo or corpo_padrao(triagem), nome), fonte
