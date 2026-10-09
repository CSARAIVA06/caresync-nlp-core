"""Extração (NER por regras) dos marcadores de um laudo laboratorial em texto livre.

Mesma estratégia de `extractMarkers` em app/lib/triage.js do site: cada linha é
normalizada, os nomes de marcadores são localizados e cada um recebe o primeiro
número entre o seu nome e o próximo marcador da linha.
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.knowledge_base import MARCADORES, Faixa, Marcador

# Aceita "6,4", "6.4", "150.000" e "1.234,5".
_NUM = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
NUMERO_RE = re.compile(_NUM)

_REF_PREFIXO = r"\b(?:vr|ref|referencia|valores? de referencia)\b\s*:?\s*"
REFERENCIA_RE = re.compile(
    _REF_PREFIXO
    + rf"(?:(?:<|≤|ate)\s*(?P<max_only>{_NUM})"
    + rf"|(?:>|≥|acima de)\s*(?P<min_only>{_NUM})"
    + rf"|(?P<min>{_NUM})\s*(?:a|-|ate)\s*(?P<max>{_NUM}))"
)

_LIMPEZA = (
    re.compile(r"\([^)]*\)"),
    re.compile(r"\[[^\]]*\]"),
    re.compile(_REF_PREFIXO + r"[<>≤≥]?\s*[\d.,]+(\s*(a|-|ate)\s*[\d.,]+)?"),
    re.compile(r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b"),
)

_PADROES_ALIAS = [
    (m, re.compile(rf"(^|[^a-z])({re.escape(alias)})(?![a-z])"))
    for m in MARCADORES
    for alias in m.aliases
]

TITULOS_EXAME = (
    (re.compile(r"hemograma"), "Hemograma"),
    (re.compile(r"perfil lipidico|lipidograma"), "Perfil lipídico"),
    (re.compile(r"perfil metabolico"), "Perfil metabólico"),
    (re.compile(r"eletrolitos|ionograma"), "Eletrólitos"),
    (re.compile(r"funcao renal"), "Função renal"),
    (re.compile(r"tireoide|tireoidiana"), "Função tireoidiana"),
    (re.compile(r"bioquimica"), "Bioquímica"),
    (re.compile(r"glicemia"), "Glicemia"),
)
HEMOGRAMA = {"hemoglobina", "hematocrito", "leucocitos", "plaquetas"}
LIPIDIOS = {"colesterol", "triglicerideos"}


def normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def parse_numero(texto: str, resto: str = "") -> float:
    """Converte números no formato brasileiro: "6,8" -> 6.8, "150.000" -> 150000, "250 mil" -> 250000."""
    if "," in texto:
        valor = float(texto.replace(".", "").replace(",", "."))
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", texto):
        valor = float(texto.replace(".", ""))
    else:
        valor = float(texto)
    if re.match(r"\s*mil\b", resto):
        valor *= 1000
    return valor


def _limpar_segmento(seg: str) -> str:
    """Remove do trecho o que não é o resultado: intervalos de referência e datas."""
    for padrao in _LIMPEZA:
        seg = padrao.sub(" ", seg)
    return seg


def _encontrar_mencoes(norm: str) -> List[Tuple[Optional[Marcador], int, int]]:
    """Menções de marcadores na linha, em ordem. Menções excluídas
    (ex.: "hemoglobina glicada") ficam com marcador None e servem só de fronteira."""
    hits = []
    for marcador, padrao in _PADROES_ALIAS:
        for r in padrao.finditer(norm):
            inicio, fim = r.start(2), r.end(2)
            excluido = marcador.exclude is not None and marcador.exclude.match(norm[fim:])
            hits.append((None if excluido else marcador, inicio, fim))
    hits.sort(key=lambda h: (h[1], -(h[2] - h[1])))

    saida, ultimo_fim = [], -1
    for h in hits:
        if h[1] < ultimo_fim:
            continue
        saida.append(h)
        ultimo_fim = h[2]
    return saida


def _mesma_escala(ref: Faixa, base: Faixa) -> bool:
    """Evita usar uma referência do laudo em outra unidade (ex.: "4,0 a 11,0 mil")."""
    pares = [(r, b) for r, b in zip(ref, base) if r is not None and b is not None]
    return bool(pares) and all(b == 0 or 0.2 <= r / b <= 5 for r, b in pares)


def _extrair_referencia(seg: str, base: Faixa) -> Optional[Faixa]:
    match = REFERENCIA_RE.search(seg)
    if not match:
        return None
    g = {k: parse_numero(v) if v else None for k, v in match.groupdict().items()}
    if g["min"] is not None:
        ref = (g["min"], g["max"])
        if ref[0] > ref[1]:
            return None
    elif g["max_only"] is not None:
        ref = (None, g["max_only"])
    else:
        ref = (g["min_only"], None)
    return ref if _mesma_escala(ref, base) else None


class LabParser:
    """Extrai informações numéricas e textuais de laudos laboratoriais brutos."""

    @staticmethod
    def parse_texto(texto: str) -> List[Dict[str, Any]]:
        resultados = []
        vistos = set()
        for linha in texto.splitlines():
            norm = normalizar(linha)
            mencoes = _encontrar_mencoes(norm)
            for i, (marcador, _, fim) in enumerate(mencoes):
                if marcador is None or marcador.id in vistos:
                    continue
                proximo = mencoes[i + 1][1] if i + 1 < len(mencoes) else len(norm)
                bruto = norm[fim:proximo]
                seg = _limpar_segmento(bruto)
                match = NUMERO_RE.search(seg)
                if not match:
                    continue
                valor = parse_numero(match.group(), seg[match.end():])
                ref_laudo = _extrair_referencia(bruto, marcador.ref)
                vistos.add(marcador.id)
                resultados.append({
                    "marcador": marcador,
                    "valor": valor,
                    "ref": ref_laudo or marcador.ref,
                    "ref_fonte": "laudo" if ref_laudo else "base",
                })
        return resultados

    @staticmethod
    def detectar_tipo_exame(texto: str, ids: List[str]) -> str:
        """Título reconhecido no laudo; senão, inferido pelos marcadores."""
        norm = normalizar(texto)
        melhor = None
        for padrao, rotulo in TITULOS_EXAME:
            m = padrao.search(norm)
            if m and (melhor is None or m.start() < melhor[0]):
                melhor = (m.start(), rotulo)
        if melhor:
            return melhor[1]
        if ids and all(i in HEMOGRAMA for i in ids):
            return "Hemograma"
        if ids and all(i in LIPIDIOS for i in ids):
            return "Perfil lipídico"
        return "Exames laboratoriais"
