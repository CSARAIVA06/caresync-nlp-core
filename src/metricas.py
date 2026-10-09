"""Métricas de avaliação do CareSync (extração, classificação e geração de texto)."""
import re
from typing import Dict, Iterable, List, Sequence, Tuple

from src.parser import parse_numero

ROTULOS_RISCO = ["NORMAL", "ATENCAO", "CRITICO"]


# ----------------------------------------------------------- extração ---

def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def comparar_extracoes(ouro: Iterable[Tuple[str, float]], previsto: Iterable[Tuple[str, float]], tol: float = 1e-6):
    """Compara pares (marcador, valor). Valor errado conta como FP e FN, como em NER estrito."""
    ouro, previsto = dict(ouro), dict(previsto)
    tp = sum(1 for k, v in previsto.items() if k in ouro and abs(ouro[k] - v) <= tol * max(1, abs(v)))
    return tp, len(previsto) - tp, len(ouro) - tp


# -------------------------------------------------------- classificação ---

def matriz_confusao(y_true: Sequence[str], y_pred: Sequence[str], rotulos=ROTULOS_RISCO) -> List[List[int]]:
    idx = {r: i for i, r in enumerate(rotulos)}
    m = [[0] * len(rotulos) for _ in rotulos]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m


def relatorio_classificacao(y_true: Sequence[str], y_pred: Sequence[str], rotulos=ROTULOS_RISCO) -> Dict:
    por_classe = {}
    for r in rotulos:
        tp = sum(t == r and p == r for t, p in zip(y_true, y_pred))
        fp = sum(t != r and p == r for t, p in zip(y_true, y_pred))
        fn = sum(t == r and p != r for t, p in zip(y_true, y_pred))
        por_classe[r] = prf(tp, fp, fn)
    acuracia = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return {
        "acuracia": acuracia,
        "f1_macro": sum(c["f1"] for c in por_classe.values()) / len(rotulos),
        "por_classe": por_classe,
        "matriz_confusao": matriz_confusao(y_true, y_pred, rotulos),
        # Erro mais grave na saúde: um caso crítico classificado como não crítico.
        "criticos_perdidos": sum(t == "CRITICO" and p != "CRITICO" for t, p in zip(y_true, y_pred)),
    }


# ---------------------------------------------------- geração de texto ---

_PALAVRA_RE = re.compile(r"\w+", re.UNICODE)
_NUMERO_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _tokens(texto: str) -> List[str]:
    # Tokenizador próprio: o do pacote rouge_score descarta letras acentuadas.
    return _PALAVRA_RE.findall(texto.lower())


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    anterior = [0] * (len(b) + 1)
    for x in a:
        atual = [0]
        for j, y in enumerate(b):
            atual.append(anterior[j] + 1 if x == y else max(anterior[j + 1], atual[j]))
        anterior = atual
    return anterior[-1]


def rouge_l(referencia: str, gerado: str) -> float:
    """F1 do ROUGE-L (maior subsequência comum de palavras)."""
    ref, ger = _tokens(referencia), _tokens(gerado)
    if not ref or not ger:
        return 0.0
    lcs = _lcs(ref, ger)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(ger), lcs / len(ref)
    return 2 * p * r / (p + r)


_VOGAIS_FORTES = set("aeoáéóâêô")


def _silabas(palavra: str) -> int:
    """Estimativa de sílabas em português: grupos de vogais, separando hiatos comuns.

    Ditongos (ex.: "ai", "ão", "ue" em "que") contam 1. Duas vogais fortes seguidas
    ("poema", "teatro", "área") ou í/ú acentuados ("saída", "baú") contam 2.
    """
    n = 0
    for g in re.findall(r"[aeiouáéíóúâêôãõàü]+", palavra.lower()):
        n += 1
        for anterior, atual in zip(g, g[1:]):
            if (anterior in _VOGAIS_FORTES and atual in _VOGAIS_FORTES) or atual in "íú":
                n += 1
    return max(1, n)


def flesch_ptbr(texto: str) -> float:
    """Índice de Flesch adaptado ao português (Martins et al., 1996).

    248,835 − 1,015 × (palavras/frases) − 84,6 × (sílabas/palavras).
    Faixas usuais: 75–100 muito fácil; 50–75 fácil; 25–50 difícil; 0–25 muito difícil.
    """
    palavras = [w for w in _PALAVRA_RE.findall(texto) if not w.isdigit()]
    frases = [f for f in re.split(r"[.!?;\n]+", texto) if _PALAVRA_RE.search(f)]
    if not palavras or not frases:
        return 0.0
    silabas = sum(_silabas(w) for w in palavras)
    return 248.835 - 1.015 * (len(palavras) / len(frases)) - 84.6 * (silabas / len(palavras))


def numeros(texto: str) -> List[float]:
    """Valores numéricos citados no texto, já normalizados ("245.000", "245 mil" e "245000" -> 245000)."""
    return [parse_numero(m.group(), texto[m.end():]) for m in _NUMERO_RE.finditer(texto)]


def fidelidade_numerica(entrada: str, gerado: str) -> bool:
    """True se todo número citado no texto gerado aparece na entrada (sem números inventados).

    Compara valores, não a grafia: "12" e "12,0", ou "245.000" e "245 mil", são o mesmo número.
    """
    disponiveis = numeros(entrada)
    return all(any(abs(n - d) <= 1e-9 * max(1.0, abs(d)) for d in disponiveis) for n in numeros(gerado))
