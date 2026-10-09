"""Linha de base: avalia o parser de regex + motor de regras no dataset sintético.

Mede a extração de pares (marcador, valor) e o nível de risco final do pipeline
regex → regras, nos splits "teste" (formatos vistos) e "teste_ood" (formatos novos).

Uso:
    python scripts/avaliar_regex.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.metricas import comparar_extracoes, prf, relatorio_classificacao  # noqa: E402
from src.parser import LabParser  # noqa: E402
from src.rules import MotorDeRegras  # noqa: E402


def triagem_regex(texto):
    dados = LabParser.parse_texto(texto)
    if not dados:
        return [], "NORMAL"  # nada extraído: o pipeline não teria como alertar
    r = MotorDeRegras.avaliar_triagem_completa(dados)
    return [(d["marcador"].id, d["valor"]) for d in dados], r.classificacao_geral.value


def main():
    laudos = [json.loads(l) for l in open(RAIZ / "data" / "laudos_sinteticos.jsonl", encoding="utf-8")]
    resultados = {}
    for split in ("teste", "teste_ood"):
        sub = [l for l in laudos if l["split"] == split]
        total = [0, 0, 0]
        por_layout = defaultdict(lambda: [0, 0, 0])
        y_true, y_pred = [], []
        for l in sub:
            ouro = [(i["id"], i["valor"]) for i in l["itens"]]
            previsto, risco = triagem_regex(l["texto"])
            c = comparar_extracoes(ouro, previsto)
            for k in range(3):
                total[k] += c[k]
                por_layout[l["layout"]][k] += c[k]
            y_true.append(l["risco"])
            y_pred.append(risco)
        resultados[split] = {
            "n_laudos": len(sub),
            "extracao": prf(*total),
            "extracao_por_layout": {k: prf(*v) for k, v in sorted(por_layout.items())},
            "triagem": relatorio_classificacao(y_true, y_pred),
        }

    saida = RAIZ / "resultados" / "baseline_regex.json"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

    for split, r in resultados.items():
        e, t = r["extracao"], r["triagem"]
        print(f"\n== {split} ({r['n_laudos']} laudos)")
        print(f"Extração  P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f}")
        for lay, m in r["extracao_por_layout"].items():
            print(f"  {lay:20s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        print(f"Triagem   acurácia={t['acuracia']:.3f} F1-macro={t['f1_macro']:.3f} "
              f"recall CRITICO={t['por_classe']['CRITICO']['recall']:.3f} críticos perdidos={t['criticos_perdidos']}")
        print(f"  matriz (linhas=real N/A/C, colunas=previsto): {t['matriz_confusao']}")
    print(f"\nSalvo em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
