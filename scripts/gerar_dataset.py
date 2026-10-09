"""Gera laudos laboratoriais sintéticos anotados para treinar e avaliar os modelos do CareSync.

Cada laudo traz, além do texto:
- entidades com posição exata (EXAME, VALOR, UNIDADE, REFERENCIA) para treinar NER;
- os resultados estruturados e o nível de risco (rótulo do classificador);
- a entrada estruturada e a explicação de rascunho para o fine-tuning do LLM.

Os laudos usam 8 formatos. Dois deles ("tabela_ref_antes" e "multilinha") só aparecem
no split "teste_ood", para medir a generalização para formatos nunca vistos no treino.

Uso:
    python scripts/gerar_dataset.py --n 1200 --seed 42
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.gerador_local import entrada_seq2seq  # noqa: E402
from src.knowledge_base import POR_ID, formatar_numero  # noqa: E402
from src.llm_service import contexto_rag  # noqa: E402
from src.mensagens import corpo_padrao, faixa_item  # noqa: E402
from src.rules import MotorDeRegras  # noqa: E402
from src.schemas import NivelRisco  # noqa: E402

GRUPOS = [
    ("HEMOGRAMA COMPLETO", "Hemograma", ["hemoglobina", "hematocrito", "leucocitos", "plaquetas"]),
    ("BIOQUÍMICA", "Bioquímica", ["glicose", "creatinina", "ureia"]),
    ("ELETRÓLITOS", "Eletrólitos", ["potassio", "sodio"]),
    ("PERFIL LIPÍDICO", "Perfil lipídico", ["colesterol", "triglicerideos"]),
    ("FUNÇÃO TIREOIDIANA", "Função tireoidiana", ["tsh"]),
]

NOMES = {
    "hemoglobina": ["Hemoglobina", "HEMOGLOBINA", "Hb", "HGB"],
    "hematocrito": ["Hematócrito", "HEMATÓCRITO", "Ht", "HCT"],
    "leucocitos": ["Leucócitos", "LEUCÓCITOS", "Leucócitos totais", "Glóbulos brancos", "WBC"],
    "plaquetas": ["Plaquetas", "PLAQUETAS", "Contagem de plaquetas", "PLT"],
    "glicose": ["Glicose", "GLICOSE", "Glicose em jejum", "Glicemia de jejum"],
    "potassio": ["Potássio", "POTÁSSIO", "Potássio sérico", "K+"],
    "sodio": ["Sódio", "SÓDIO", "Sódio sérico", "Na+"],
    "creatinina": ["Creatinina", "CREATININA", "Creatinina sérica"],
    "ureia": ["Ureia", "URÉIA", "Uréia"],
    "colesterol": ["Colesterol total", "COLESTEROL TOTAL", "Colesterol Total"],
    "triglicerideos": ["Triglicerídeos", "TRIGLICERÍDEOS", "Triglicérides"],
    "tsh": ["TSH", "TSH ultrassensível", "Hormônio tireoestimulante (TSH)"],
}

UNIDADES = {
    "g/dL": ["g/dL", "g/dl"],
    "%": ["%"],
    "/mm³": ["/mm³", "/mm3", "/µL"],
    "mg/dL": ["mg/dL", "mg/dl"],
    "mEq/L": ["mEq/L", "mmol/L"],
    "µUI/mL": ["µUI/mL", "uUI/mL", "mUI/L"],
}

# Casas decimais usadas pelos laboratórios (negativo = arredonda centenas/milhares).
CASAS = {
    "hemoglobina": 1, "hematocrito": 0, "leucocitos": -2, "plaquetas": -3, "glicose": 0,
    "potassio": 1, "sodio": 0, "creatinina": 2, "ureia": 0, "colesterol": 0,
    "triglicerideos": 0, "tsh": 2,
}

# Referências alternativas impressas por alguns laboratórios (ex.: por sexo).
REF_VARIANTES = {
    "hemoglobina": [(12.0, 15.5), (13.5, 17.5)],
    "hematocrito": [(36, 46), (40, 52)],
    "creatinina": [(0.5, 1.1), (0.7, 1.3)],
}

LABORATORIOS = ["Laboratório São Lucas", "LabVida Análises Clínicas", "Centro Diagnóstico Jundiaí", "Laboratório Bio Saúde"]
PACIENTES = ["Ana Souza", "José Almeida", "Maria Oliveira", "Carlos Pereira", "Luiza Martins", "Roberto Lima",
             "Fernanda Costa", "Paulo Ribeiro", "Juliana Rocha", "Marcos Teixeira", "Beatriz Gomes", "Rafael Dias"]

LAYOUTS_VISTOS = ["dois_pontos_vr", "referencia_extenso", "pontilhado", "sem_referencia", "tabela_pipe", "narrativo"]
LAYOUTS_OOD = ["tabela_ref_antes", "multilinha"]


class Texto:
    """Monta o laudo registrando a posição exata de cada entidade."""

    def __init__(self):
        self.partes, self.n, self.entidades = [], 0, []

    def add(self, s, label=None, item=None):
        if label:
            self.entidades.append({"inicio": self.n, "fim": self.n + len(s), "label": label, "texto": s, "item": item})
        self.partes.append(s)
        self.n += len(s)

    def nl(self):
        self.add("\n")

    def valor(self):
        return "".join(self.partes)


# ----------------------------------------------------------------- valores ---

def _sortear_valor(rng, m, alvo):
    """Sorteia um valor do marcador tentando cair no nível alvo (o rótulo final é recalculado)."""
    ref_min, ref_max = m.ref
    crit_min, crit_max = m.critico
    if alvo == NivelRisco.NORMAL:
        lo = ref_min if ref_min is not None else ref_max * 0.6
        return rng.uniform(lo, ref_max)
    if alvo == NivelRisco.ATENCAO:
        lados = []
        if ref_min is not None:
            lados.append((crit_min if crit_min is not None else ref_min * 0.7, ref_min))
        if ref_max is not None:
            lados.append((ref_max, crit_max if crit_max is not None else ref_max * 1.4))
        return rng.uniform(*rng.choice(lados))
    lados = []
    if crit_min is not None:
        lados.append((crit_min * 0.6, crit_min))
    if crit_max is not None:
        lados.append((crit_max, crit_max * 1.3))
    return rng.uniform(*rng.choice(lados))


def _arredondar(marcador_id, v):
    casas = CASAS[marcador_id]
    v = round(v, casas)
    return float(v) if casas > 0 else float(int(v))


def _formatar_valor(rng, marcador_id, v):
    """Texto do valor como apareceria no laudo: "13,5", "13.5", "6.800", "6800", "6,8 mil"."""
    casas = CASAS[marcador_id]
    if casas > 0:
        texto = f"{v:.{casas}f}"
        return texto.replace(".", ",") if rng.random() < 0.8 else texto
    if v >= 1000:
        estilo = rng.choice(["br", "br", "us", "mil"])
        if estilo == "mil":
            mil = v / 1000
            texto = f"{mil:.1f}".replace(".", ",") if mil % 1 else f"{int(mil)}"
            return f"{texto} mil"
        if estilo == "br":
            return f"{int(v):,}".replace(",", ".")
        return str(int(v))
    return str(int(v))


def _formatar_ref(rng, ref):
    lo, hi = ref

    def num(x):
        return f"{x:,.2f}".rstrip("0").rstrip(".").replace(",", "_").replace(".", ",").replace("_", ".")

    if lo is None:
        return rng.choice([f"< {num(hi)}", f"até {num(hi)}", f"inferior a {num(hi)}"])
    return f"{num(lo)} {rng.choice(['a', '-', 'até'])} {num(hi)}"


# ----------------------------------------------------------------- layouts ---

def _cel(t, s, label, item):
    t.add(s, label, item)


def render_dois_pontos_vr(t, itens, rng):
    for k, it in enumerate(itens):
        _cel(t, it["nome"], "EXAME", it["idx"]); t.add(": ")
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"]); t.add(" " * rng.randint(3, 8) + "(VR: ")
        _cel(t, it["ref_txt"], "REFERENCIA", it["idx"]); t.add(")"); t.nl()


def render_referencia_extenso(t, itens, rng):
    for it in itens:
        _cel(t, it["nome"], "EXAME", it["idx"]); t.add(": ")
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"]); t.add(" " * rng.randint(2, 6) + "Referência: ")
        _cel(t, it["ref_txt"], "REFERENCIA", it["idx"]); t.nl()


def render_pontilhado(t, itens, rng):
    for it in itens:
        _cel(t, it["nome"].upper(), "EXAME", it["idx"]); t.add("." * rng.randint(3, 15) + ": ")
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"]); t.add("   Valores de referência: ")
        _cel(t, it["ref_txt"], "REFERENCIA", it["idx"]); t.nl()


def render_sem_referencia(t, itens, rng):
    for it in itens:
        _cel(t, it["nome"], "EXAME", it["idx"]); t.add(rng.choice([" ", ": ", " - "]))
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"]); t.nl()


def render_tabela_pipe(t, itens, rng):
    t.add("Exame | Resultado | Unidade | Referência"); t.nl()
    for it in itens:
        _cel(t, it["nome"], "EXAME", it["idx"]); t.add(" | ")
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" | ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"]); t.add(" | ")
        _cel(t, it["ref_txt"], "REFERENCIA", it["idx"]); t.nl()


def render_narrativo(t, itens, rng):
    t.add(rng.choice(["Resultado: ", "Paciente apresenta ", "Observa-se "]))
    for k, it in enumerate(itens):
        if k:
            t.add(" e " if k == len(itens) - 1 else ", ")
        nome = it["nome"] if it["nome"].isupper() or it["nome"][1:2].isupper() else it["nome"].lower()
        _cel(t, nome, "EXAME", it["idx"]); t.add(" de ")
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"])
    t.add("."); t.nl()


def render_tabela_ref_antes(t, itens, rng):
    t.add("Exame | Referência | Resultado"); t.nl()
    for it in itens:
        _cel(t, it["nome"], "EXAME", it["idx"]); t.add(" | ")
        _cel(t, it["ref_txt"], "REFERENCIA", it["idx"]); t.add(" | ")
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"]); t.nl()


def render_multilinha(t, itens, rng):
    for it in itens:
        _cel(t, it["nome"].upper(), "EXAME", it["idx"]); t.nl()
        t.add("  Resultado: ")
        _cel(t, it["valor_txt"], "VALOR", it["idx"]); t.add(" ")
        _cel(t, it["unidade"], "UNIDADE", it["idx"]); t.nl()
        t.add("  Valor de referência: ")
        _cel(t, it["ref_txt"], "REFERENCIA", it["idx"]); t.nl()


RENDER = {nome: globals()[f"render_{nome}"] for nome in LAYOUTS_VISTOS + LAYOUTS_OOD}
SEM_REFERENCIA = {"sem_referencia", "narrativo"}


# ----------------------------------------------------- textos do seq2seq ---
# Várias formulações por nível e por item, para o modelo aprender a explicar
# (e não só copiar um único template). Nenhuma afirma diagnóstico.

INTROS = {
    NivelRisco.NORMAL: [
        "Seus resultados chegaram e temos uma boa notícia: todos os exames analisados estão dentro dos valores esperados.",
        "Boa notícia: os exames analisados estão dentro da faixa considerada adequada.",
        "Analisamos seus exames e todos os valores estão dentro do esperado.",
    ],
    NivelRisco.ATENCAO: [
        "Seus resultados chegaram. A maior parte está dentro do esperado, mas alguns valores merecem a atenção do seu médico. Isso é comum e não indica uma emergência.",
        "Alguns valores do seu exame ficaram fora da faixa de referência. Isso é comum e não indica uma emergência, mas vale conversar com seu médico.",
        "Seus exames mostram pequenas alterações que merecem acompanhamento médico, sem caráter de urgência.",
    ],
    NivelRisco.CRITICO: [
        "Seus resultados chegaram e um ou mais valores precisam de avaliação médica rápida. Nossa equipe de saúde já foi avisada e vai entrar em contato com você.",
        "Um ou mais valores do seu exame precisam ser avaliados rapidamente. Nossa equipe de saúde já recebeu um aviso prioritário.",
        "Identificamos valores que precisam de avaliação médica com prioridade. A equipe de enfermagem já foi notificada e entrará em contato.",
    ],
}

ITEM_NORMAL = [
    "• {nome}: {valor} {unidade}. É a {explicacao}, e está normal.",
    "• {nome} ({valor} {unidade}) está dentro da referência.",
    "• {nome}: {valor} {unidade}, dentro do esperado.",
]

ITEM_ALTERADO = [
    "• {nome}: {valor} {unidade} ({lado} da referência de {faixa}). É a {explicacao}. Esse resultado {consequencia}.",
    "• {nome} está {lado} do esperado ({valor} {unidade}; referência de {faixa}). Esse exame mede a {explicacao}, e o resultado {consequencia}.",
    "• O resultado de {nome_min} foi {valor} {unidade}, {lado} da faixa de {faixa}. Trata-se da {explicacao}; um valor assim {consequencia}.",
]


def _campos_item(item):
    m = POR_ID[item.id]
    nome_min = item.parametro if item.parametro.isupper() else item.parametro[0].lower() + item.parametro[1:]
    return {
        "nome": item.parametro, "nome_min": nome_min, "valor": formatar_numero(item.valor), "unidade": item.unidade,
        "faixa": faixa_item(item), "explicacao": m.explicacao,
        "lado": "abaixo" if item.direcao == "baixo" else "acima",
        "consequencia": m.baixo if item.direcao == "baixo" else m.alto,
    }


def explicacao_variada(rng, triagem):
    nivel = triagem.classificacao_geral
    if nivel == NivelRisco.NORMAL:
        modelo = rng.choice(ITEM_NORMAL)
        linhas = [modelo.format(**_campos_item(i)) for i in triagem.exames_extraidos]
    else:
        modelo = rng.choice(ITEM_ALTERADO)
        linhas = [modelo.format(**_campos_item(i)) for i in triagem.exames_extraidos if i.status_item != NivelRisco.NORMAL]
    return "\n".join([rng.choice(INTROS[nivel]), *linhas])


# --------------------------------------------------------------- NER (BIO) ---

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|\w+|[^\w\s]")


def tokens_bio(texto, entidades):
    """Tokeniza o laudo e marca cada token com B-/I-/O a partir das posições das entidades."""
    tokens, tags = [], []
    for t in TOKEN_RE.finditer(texto):
        tag = "O"
        for e in entidades:
            if e["inicio"] <= t.start() and t.end() <= e["fim"]:
                tag = ("B-" if t.start() == e["inicio"] else "I-") + e["label"]
                break
        tokens.append(t.group())
        tags.append(tag)
    return tokens, tags


# ------------------------------------------------------------------- laudo ---

def gerar_laudo(rng, i, layout, alvo):
    # Escolhe os grupos garantindo que o "marcador principal" do nível alvo esteja presente.
    candidatos = [mid for _, _, ids in GRUPOS for mid in ids
                  if alvo != NivelRisco.CRITICO or any(c is not None for c in POR_ID[mid].critico)]
    principal = rng.choice(candidatos)
    grupos = [g for g in GRUPOS if principal in g[2]]
    outros = [g for g in GRUPOS if principal not in g[2]]
    grupos += rng.sample(outros, rng.choice([0, 0, 1, 1, 2]))
    rng.shuffle(grupos)

    t = Texto()
    t.add(rng.choice(LABORATORIOS)); t.nl()
    t.add(f"Paciente: {rng.choice(PACIENTES)}"); t.nl()
    t.add(f"Data da coleta: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026"); t.nl()
    t.add("Material: Sangue"); t.nl(); t.nl()

    itens = []
    for titulo, _, ids in grupos:
        presentes = [mid for mid in ids if mid == principal or rng.random() < 0.75] or [ids[0]]
        bloco = []
        for mid in presentes:
            m = POR_ID[mid]
            if mid == principal:
                nivel = alvo
            elif alvo != NivelRisco.NORMAL and rng.random() < 0.25:
                nivel = NivelRisco.ATENCAO
            else:
                nivel = NivelRisco.NORMAL
            valor = _arredondar(mid, _sortear_valor(rng, m, nivel))
            if layout in SEM_REFERENCIA:
                ref, ref_fonte = m.ref, "base"
            else:
                ref = rng.choice(REF_VARIANTES[mid]) if mid in REF_VARIANTES and rng.random() < 0.3 else m.ref
                ref_fonte = "laudo"
            item = {
                "idx": len(itens), "id": mid, "nome": rng.choice(NOMES[mid]), "valor": valor,
                "valor_txt": _formatar_valor(rng, mid, valor), "unidade": rng.choice(UNIDADES[m.unidade]),
                "ref": ref, "ref_txt": _formatar_ref(rng, ref), "ref_fonte": ref_fonte,
            }
            itens.append(item)
            bloco.append(item)
        t.add(titulo); t.nl()
        RENDER[layout](t, bloco, rng)
        t.nl()
    t.add(rng.choice(["Liberado eletronicamente.", "Resultados conferidos.", "Método: automatizado."]))

    # Rótulos: mesmas regras da API, aplicadas aos valores verdadeiros (não ao texto).
    dados = [{"marcador": POR_ID[it["id"]], "valor": it["valor"], "ref": it["ref"], "ref_fonte": it["ref_fonte"]}
             for it in itens]
    triagem = MotorDeRegras.avaliar_triagem_completa(dados, grupos[0][1])
    texto = t.valor()
    assert all(texto[e["inicio"]:e["fim"]] == e["texto"] for e in t.entidades)

    return {
        "id": f"L{i:04d}",
        "layout": layout,
        "texto": texto,
        "entidades": t.entidades,
        "itens": [e.model_dump(mode="json") for e in triagem.exames_extraidos],
        "tipo_exame": triagem.tipo_exame,
        "risco": triagem.classificacao_geral.value,
        "entrada_llm": f"Classificação geral do exame: {triagem.classificacao_geral.value}\n"
                       f"Tipo de exame: {triagem.tipo_exame}\n"
                       f"Resultados e base de conhecimento:\n{contexto_rag(triagem)}",
        "explicacao_padrao": corpo_padrao(triagem),
        "entrada_seq2seq": entrada_seq2seq(triagem),
        "explicacao_alvo": explicacao_variada(rng, triagem),
        # Preencha com o texto revisado por uma pessoa (de preferência da área da saúde).
        "explicacao_revisada": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", type=Path, default=RAIZ / "data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    alvos = [NivelRisco.NORMAL, NivelRisco.ATENCAO, NivelRisco.CRITICO]
    layouts = LAYOUTS_VISTOS + LAYOUTS_OOD
    laudos = [gerar_laudo(rng, i, layouts[i % len(layouts)], alvos[(i // len(layouts)) % 3]) for i in range(args.n)]

    vistos = [l for l in laudos if l["layout"] in LAYOUTS_VISTOS]
    rng.shuffle(vistos)
    n_treino, n_val = int(len(vistos) * 0.7), int(len(vistos) * 0.15)
    for k, laudo in enumerate(vistos):
        laudo["split"] = "treino" if k < n_treino else "validacao" if k < n_treino + n_val else "teste"
    for laudo in laudos:
        laudo.setdefault("split", "teste_ood")

    args.saida.mkdir(parents=True, exist_ok=True)

    def jsonl(nome, linhas):
        with open(args.saida / nome, "w", encoding="utf-8") as f:
            for linha in linhas:
                f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    # Base completa (todas as anotações de cada laudo)
    jsonl("laudos_sinteticos.jsonl", laudos)

    # Recorte 1: NER em formato BIO (tokens + tags)
    jsonl("ner_bio.jsonl", (
        {"id": l["id"], "split": l["split"], "layout": l["layout"],
         **dict(zip(("tokens", "tags"), tokens_bio(l["texto"], l["entidades"])))}
        for l in laudos
    ))

    # Recorte 2: classificação de risco (texto do laudo -> rótulo)
    with open(args.saida / "classificacao.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "layout", "risco", "texto"])
        for l in laudos:
            w.writerow([l["id"], l["split"], l["layout"], l["risco"], l["texto"]])

    # Recorte 3: seq2seq para o PTT5 (resultados estruturados -> explicação)
    jsonl("seq2seq.jsonl", (
        {"id": l["id"], "split": l["split"], "risco": l["risco"],
         "input_text": l["entrada_seq2seq"], "target_text": l["explicacao_revisada"] or l["explicacao_alvo"]}
        for l in laudos
    ))

    print(f"{len(laudos)} laudos em {args.saida}")
    print("split:", dict(Counter(l["split"] for l in laudos)))
    print("risco:", dict(Counter(l["risco"] for l in laudos)))
    print("layout:", dict(Counter(l["layout"] for l in laudos)))
    print("entidades:", dict(Counter(e["label"] for l in laudos for e in l["entidades"])))


if __name__ == "__main__":
    main()
