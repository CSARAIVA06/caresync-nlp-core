"""Gerador local (offline) da explicação ao paciente com PTT5 ajustado (seq2seq).

A entrada é o resultado já estruturado e classificado pelas regras; o modelo só
"traduz" para linguagem simples. Assim, um erro do modelo nunca altera o nível de risco.

Requer torch e transformers (requirements-ml.txt); importados sob demanda.
"""
from src.knowledge_base import formatar_numero
from src.mensagens import faixa_item
from src.schemas import ResultadoTriagem

PREFIXO = "triagem"


def entrada_seq2seq(triagem: ResultadoTriagem) -> str:
    """Entrada compacta: nível de risco + resultados estruturados. Igual no treino e na inferência."""
    partes = [f"{PREFIXO}: {triagem.classificacao_geral.value}"]
    for i in triagem.exames_extraidos:
        situacao = {"baixo": "baixo", "alto": "alto"}.get(i.direcao, "normal")
        partes.append(f"{i.parametro}: {formatar_numero(i.valor)} {i.unidade} (ref {faixa_item(i)}; {situacao})")
    return " | ".join(partes)


class GeradorPTT5:
    def __init__(self, pasta_modelo: str, max_input: int = 512, max_new_tokens: int = 384, num_beams: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(pasta_modelo)
        self.modelo = AutoModelForSeq2SeqLM.from_pretrained(pasta_modelo).eval()
        self.max_input = max_input
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def gerar_textos(self, entradas):
        enc = self.tokenizer(entradas, max_length=self.max_input, truncation=True, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            saida = self.modelo.generate(**enc, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams)
        return self.tokenizer.batch_decode(saida, skip_special_tokens=True)

    def gerar(self, triagem: ResultadoTriagem) -> str:
        return self.gerar_textos([entrada_seq2seq(triagem)])[0].strip()
