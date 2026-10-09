"""Extrator de marcadores baseado em NER (BERTimbau ajustado para Token Classification).

Substituto direto do LabParser: `parse_texto` devolve a mesma estrutura, então o motor
de regras e a API não mudam. O modelo marca os trechos EXAME, VALOR, UNIDADE e
REFERENCIA; este módulo agrupa cada exame com o valor e a referência que vêm depois
dele e converte os números (as regras continuam fazendo a matemática).

Requer torch e transformers (requirements-ml.txt). Os imports são feitos sob demanda
para que a API continue leve quando o extrator de regex estiver em uso.
"""
from typing import Any, Dict, List, Optional, Tuple

from src.parser import NUMERO_RE, identificar_marcador, parse_faixa, parse_numero

ENTIDADES = ["EXAME", "VALOR", "UNIDADE", "REFERENCIA"]
LABELS = ["O"] + [f"{p}-{e}" for e in ENTIDADES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))

Span = Tuple[str, int, int]  # (entidade, início, fim) em caracteres


def rotular_tokens(offsets, entidades) -> List[int]:
    """Converte entidades (posições em caracteres) em rótulos BIO por subpalavra; -100 = ignorar."""
    rotulos = []
    for inicio, fim in offsets:
        if inicio == fim:  # tokens especiais ([CLS], [SEP], padding)
            rotulos.append(-100)
            continue
        rotulo = "O"
        for e in entidades:
            if e["inicio"] <= inicio and fim <= e["fim"]:
                rotulo = ("B-" if inicio == e["inicio"] else "I-") + e["label"]
                break
        rotulos.append(LABEL2ID[rotulo])
    return rotulos


def tokens_para_spans(offsets, ids_previstos) -> List[Span]:
    """Junta subpalavras B-/I- consecutivas da mesma entidade em trechos de texto.

    Um I- só continua a entidade se o token anterior for da mesma entidade; depois de
    um "O" (ou de outra entidade), um I- solto abre um trecho novo.
    """
    spans: List[List] = []
    anterior = None
    for (inicio, fim), lid in zip(offsets, ids_previstos):
        if inicio == fim:
            continue
        rotulo = ID2LABEL[int(lid)]
        if rotulo == "O":
            anterior = None
            continue
        prefixo, entidade = rotulo.split("-", 1)
        if prefixo == "I" and anterior == entidade:
            spans[-1][2] = fim
        else:
            spans.append([entidade, inicio, fim])
        anterior = entidade
    return [tuple(s) for s in spans]


def mesclar_janelas(janelas) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Une as previsões de janelas sobrepostas (laudos maiores que max_length).

    Cada token fica com a previsão da janela em que está mais longe da borda, onde o
    modelo viu contexto dos dois lados; assim uma entidade cortada no fim de uma janela
    (ex.: "150." de "150.000") é substituída pela versão completa da janela seguinte.
    `janelas` é uma lista de (offsets, ids_previstos).
    """
    melhor: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for offsets, previstos in janelas:
        reais = [i for i, (a, b) in enumerate(offsets) if a != b]
        for pos, i in enumerate(reais):
            chave = (int(offsets[i][0]), int(offsets[i][1]))
            distancia = min(pos, len(reais) - 1 - pos)
            if chave not in melhor or distancia > melhor[chave][0]:
                melhor[chave] = (distancia, int(previstos[i]))
    chaves = sorted(melhor)
    return chaves, [melhor[k][1] for k in chaves]


def spans_para_itens(texto: str, spans: List[Span]) -> List[Dict[str, Any]]:
    """Cada EXAME recebe o primeiro VALOR e a primeira REFERENCIA até o próximo EXAME."""
    itens, vistos = [], set()
    exames = [i for i, s in enumerate(spans) if s[0] == "EXAME"]
    for k, i in enumerate(exames):
        marcador = identificar_marcador(texto[spans[i][1]:spans[i][2]])
        if marcador is None or marcador.id in vistos:
            continue
        limite = exames[k + 1] if k + 1 < len(exames) else len(spans)
        valor: Optional[float] = None
        ref = None
        for ent, ini, fim in spans[i + 1:limite]:
            trecho = texto[ini:fim]
            if ent == "VALOR" and valor is None and (m := NUMERO_RE.search(trecho)):
                valor = parse_numero(m.group(), trecho[m.end():])
            elif ent == "REFERENCIA" and ref is None:
                ref = parse_faixa(trecho, marcador.ref)
        if valor is None:
            continue
        vistos.add(marcador.id)
        itens.append({
            "marcador": marcador,
            "valor": valor,
            "ref": ref or marcador.ref,
            "ref_fonte": "laudo" if ref else "base",
        })
    return itens


class NerExtractor:
    """Carrega um modelo de NER treinado (pasta com config, pesos e tokenizador)."""

    def __init__(self, pasta_modelo: str, max_length: int = 512):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(pasta_modelo)
        self.modelo = AutoModelForTokenClassification.from_pretrained(pasta_modelo).eval()
        self.max_length = max_length

    def prever_spans(self, texto: str) -> List[Span]:
        # Laudos maiores que a janela são processados em blocos sobrepostos.
        enc = self.tokenizer(
            texto, return_offsets_mapping=True, truncation=True, max_length=self.max_length,
            stride=64, return_overflowing_tokens=True, padding=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        with self.torch.no_grad():
            previsoes = self.modelo(**enc).logits.argmax(-1).tolist()
        return tokens_para_spans(*mesclar_janelas(zip(offsets, previsoes)))

    def parse_texto(self, texto: str) -> List[Dict[str, Any]]:
        return spans_para_itens(texto, self.prever_spans(texto))
