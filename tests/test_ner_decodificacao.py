"""Partes do NER que não dependem do modelo: rótulos BIO por subpalavra e montagem dos itens."""
import pytest

from src.ner import ID2LABEL, LABEL2ID, rotular_tokens, spans_para_itens, tokens_para_spans
from src.parser import parse_faixa
from src.knowledge_base import POR_ID

TEXTO = "Hb: 6,8 g/dL (VR: 12,0 a 17,5)\nPLT 245 mil /mm³"
#        0123456789...


def _ent(texto, trecho, label, a_partir=0):
    i = texto.index(trecho, a_partir)
    return {"inicio": i, "fim": i + len(trecho), "label": label}


ENTIDADES = [
    _ent(TEXTO, "Hb", "EXAME"), _ent(TEXTO, "6,8", "VALOR"), _ent(TEXTO, "g/dL", "UNIDADE"),
    _ent(TEXTO, "12,0 a 17,5", "REFERENCIA"),
    _ent(TEXTO, "PLT", "EXAME"), _ent(TEXTO, "245 mil", "VALOR"), _ent(TEXTO, "/mm³", "UNIDADE"),
]


def _offsets_simples(texto):
    """Simula subpalavras: cada palavra/número/pontuação vira um token, mais [CLS] e [SEP]."""
    import re
    return [(0, 0)] + [(m.start(), m.end()) for m in re.finditer(r"\d+|\w+|[^\w\s]", texto)] + [(0, 0)]


def test_rotular_tokens_bio():
    offsets = _offsets_simples(TEXTO)
    rotulos = [ID2LABEL.get(r, "IGN") for r in rotular_tokens(offsets, ENTIDADES)]
    tokens = ["[CLS]"] + [TEXTO[a:b] for a, b in offsets[1:-1]] + ["[SEP]"]
    pares = dict(zip(tokens, rotulos))
    assert rotulos[0] == rotulos[-1] == "IGN"
    assert pares["Hb"] == "B-EXAME"
    assert list(zip(tokens, rotulos))[3:6] == [("6", "B-VALOR"), (",", "I-VALOR"), ("8", "I-VALOR")]
    assert pares["mil"] == "I-VALOR"
    assert pares["VR"] == "O"


def test_ida_e_volta_entidades_para_itens():
    offsets = _offsets_simples(TEXTO)
    spans = tokens_para_spans(offsets, rotular_tokens(offsets, ENTIDADES))
    assert [(e, TEXTO[a:b]) for e, a, b in spans] == [
        ("EXAME", "Hb"), ("VALOR", "6,8"), ("UNIDADE", "g/dL"), ("REFERENCIA", "12,0 a 17,5"),
        ("EXAME", "PLT"), ("VALOR", "245 mil"), ("UNIDADE", "/mm³"),
    ]
    itens = spans_para_itens(TEXTO, spans)
    assert [(i["marcador"].id, i["valor"], i["ref_fonte"]) for i in itens] == [
        ("hemoglobina", 6.8, "laudo"), ("plaquetas", 245000.0, "base"),
    ]
    assert itens[0]["ref"] == (12.0, 17.5)


def test_exame_sem_valor_e_ignorado():
    texto = "Hemoglobina: não realizado"
    spans = [("EXAME", 0, 11)]
    assert spans_para_itens(texto, spans) == []


def test_rotulos_cobrem_todas_as_entidades():
    assert set(LABEL2ID) == {"O"} | {f"{p}-{e}" for e in ("EXAME", "VALOR", "UNIDADE", "REFERENCIA") for p in "BI"}


@pytest.mark.parametrize("texto, esperado", [
    ("12,0 a 17,5", (12.0, 17.5)),
    ("12,0 - 17,5", (12.0, 17.5)),
    ("12,0 até 17,5", (12.0, 17.5)),
    ("< 190", (None, 190.0)),
    ("inferior a 190", (None, 190.0)),
    ("4,0 a 11,0", None),  # escala diferente da base de leucócitos
    ("17,5 a 12,0", None),
])
def test_parse_faixa(texto, esperado):
    marcador = {"< 190": "colesterol", "inferior a 190": "colesterol", "4,0 a 11,0": "leucocitos"}.get(texto, "hemoglobina")
    assert parse_faixa(texto, POR_ID[marcador].ref) == esperado
