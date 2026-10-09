"""Junta os JSONs de resultados/ em uma tabela Markdown (resultados/RESULTADOS.md) para o artigo.

Uso:
    python scripts/compilar_resultados.py
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "resultados"


def carregar(nome):
    p = PASTA / nome
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def linha_triagem(sistema, split, tri, ext=None):
    f1_ext = f"{ext['f1']:.3f}" if ext else "—"
    return (f"| {sistema} | {split} | {f1_ext} | {tri['acuracia']:.3f} | {tri['f1_macro']:.3f} | "
            f"{tri['por_classe']['CRITICO']['recall']:.3f} | {tri['criticos_perdidos']} |")


def main():
    base = carregar("baseline_regex.json")
    clf = carregar("classificador.json")
    ner = carregar("ner_ner-bertimbau.json")
    ptt5 = carregar("ptt5_ptt5-caresync.json")

    out = ["# Resultados experimentais — CareSync", "",
           "Dataset sintético (seed 42): 630 laudos de treino, 135 de validação, 135 de teste "
           "(formatos vistos) e 300 de teste OOD (formatos nunca vistos no treino).", "",
           "## Extração e triagem de risco", "",
           "| Sistema | Split | F1 extração | Acurácia risco | F1-macro risco | Recall CRÍTICO | Críticos perdidos |",
           "|---|---|---|---|---|---|---|"]
    for split in ("teste", "teste_ood"):
        if base:
            out.append(linha_triagem("Regex + regras", split, base[split]["triagem"], base[split]["extracao"]))
        if ner:
            p = ner[split]["pipeline_ner_regras"]
            out.append(linha_triagem("NER (BERTimbau) + regras", split, p["triagem"], p["extracao"]))
        if clf:
            out.append(linha_triagem("TF-IDF + SVM", split, clf[split]["ml_tfidf_svm"]))
            out.append(linha_triagem("Híbrido (SVM + regras/regex)", split, clf[split]["hibrido"]))

    if ner:
        out += ["", "## NER — entidades (seqeval, micro)", "", "| Split | Precision | Recall | F1 |", "|---|---|---|---|"]
        for split in ("teste", "teste_ood"):
            e = ner[split]["entidades"]
            out.append(f"| {split} | {e[f'{split}_precision']:.3f} | {e[f'{split}_recall']:.3f} | {e[f'{split}_f1']:.3f} |")

    if ptt5:
        t = ptt5["teste"]
        out += ["", "## Geração (PTT5) — split teste", "", "| Métrica | Valor |", "|---|---|",
                f"| ROUGE-L (F1) | {t['rouge_l']:.3f} |",
                f"| Flesch PT-BR — laudo bruto | {t['flesch_laudo_bruto']:.1f} |",
                f"| Flesch PT-BR — texto gerado | {t['flesch_gerado']:.1f} |",
                f"| Flesch PT-BR — referência | {t['flesch_referencia']:.1f} |",
                f"| Fidelidade numérica | {t['fidelidade_numerica']:.3f} |",
                f"| Violações de guardrail | {t['taxa_violacao_guardrail']:.3f} |"]

    out += ["", "_Rótulos de risco gerados pelas mesmas regras determinísticas da API; por isso o motor de "
            "regras é exato quando a extração é correta. Ver seção de limitações._", ""]
    (PASTA / "RESULTADOS.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
