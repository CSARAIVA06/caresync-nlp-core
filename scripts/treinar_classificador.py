"""Fase 3: classificador de risco (TF-IDF + SVM) comparado com as regras e com a triagem híbrida.

- ML:      TF-IDF (palavras + caracteres) + LinearSVC, treinado no texto bruto do laudo.
- Regras:  parser de regex + motor de regras determinístico (pipeline atual da API).
- Híbrido: o ML decide, mas as regras sobrescrevem para CRITICO sempre que um
           limite crítico é ultrapassado (hard guardrail).

Uso:
    python scripts/treinar_classificador.py
"""
import csv
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.avaliar_regex import triagem_regex  # noqa: E402
from src.metricas import relatorio_classificacao  # noqa: E402


def carregar():
    with open(RAIZ / "data" / "classificacao.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return {s: [l for l in linhas if l["split"] == s] for s in ("treino", "validacao", "teste", "teste_ood")}


def novo_modelo(c):
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("palavras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("svm", LinearSVC(C=c, class_weight="balanced")),
    ])


def hibrido(pred_ml, pred_regras):
    return ["CRITICO" if r == "CRITICO" else m for m, r in zip(pred_ml, pred_regras)]


def main():
    dados = carregar()
    textos = {s: [l["texto"] for l in v] for s, v in dados.items()}
    rotulos = {s: [l["risco"] for l in v] for s, v in dados.items()}

    # Escolhe C pela F1-macro na validação
    melhor = None
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        modelo = novo_modelo(c).fit(textos["treino"], rotulos["treino"])
        f1 = relatorio_classificacao(rotulos["validacao"], list(modelo.predict(textos["validacao"])))["f1_macro"]
        print(f"C={c:<5} F1-macro validação={f1:.3f}")
        if melhor is None or f1 > melhor[1]:
            melhor = (c, f1, modelo)
    c, _, modelo = melhor
    print(f"Melhor C={c}")

    resultados = {"C": c}
    for split in ("teste", "teste_ood"):
        y = rotulos[split]
        pred_ml = list(modelo.predict(textos[split]))
        pred_regras = [triagem_regex(t)[1] for t in textos[split]]
        resultados[split] = {
            "ml_tfidf_svm": relatorio_classificacao(y, pred_ml),
            "regras_regex": relatorio_classificacao(y, pred_regras),
            "hibrido": relatorio_classificacao(y, hibrido(pred_ml, pred_regras)),
        }

    pasta_modelos = RAIZ / "modelos"
    pasta_modelos.mkdir(exist_ok=True)
    joblib.dump(modelo, pasta_modelos / "classificador_risco.joblib")
    saida = RAIZ / "resultados" / "classificador.json"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

    for split in ("teste", "teste_ood"):
        print(f"\n== {split}")
        print(f"{'sistema':14s} {'acurácia':>9s} {'F1-macro':>9s} {'recall CRIT':>12s} {'crít. perdidos':>15s}")
        for nome, r in resultados[split].items():
            print(f"{nome:14s} {r['acuracia']:9.3f} {r['f1_macro']:9.3f} "
                  f"{r['por_classe']['CRITICO']['recall']:12.3f} {r['criticos_perdidos']:15d}")
        print(f"  matriz ML (linhas=real N/A/C): {resultados[split]['ml_tfidf_svm']['matriz_confusao']}")
    print(f"\nSalvo em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
