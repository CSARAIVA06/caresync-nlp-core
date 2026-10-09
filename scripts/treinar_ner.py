"""Fase 2: fine-tuning do BERTimbau para NER em laudos (Token Classification).

Avalia em dois níveis, nos splits "teste" e "teste_ood":
1. entidades (seqeval): precision/recall/F1 por tipo de entidade;
2. pipeline completo NER → regras: extração (marcador, valor) e nível de risco,
   com as mesmas métricas de scripts/avaliar_regex.py (comparação direta).

Uso (GPU no Colab):
    python scripts/treinar_ner.py
Teste rápido na CPU:
    python scripts/treinar_ner.py --limite 40 --max-steps 5 --saida modelos/ner-teste
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.metricas import comparar_extracoes, prf, relatorio_classificacao  # noqa: E402
from src.ner import ID2LABEL, LABEL2ID, LABELS, NerExtractor, rotular_tokens  # noqa: E402
from src.rules import MotorDeRegras  # noqa: E402


def carregar(limite=None):
    laudos = [json.loads(l) for l in open(RAIZ / "data" / "laudos_sinteticos.jsonl", encoding="utf-8")]
    por_split = defaultdict(list)
    for l in laudos:
        por_split[l["split"]].append(l)
    if limite:
        por_split = {s: v[:limite] for s, v in por_split.items()}
    return por_split


def tokenizar(laudos, tokenizer, max_length):
    enc = tokenizer([l["texto"] for l in laudos], truncation=True, max_length=max_length,
                    return_offsets_mapping=True)
    enc["labels"] = [rotular_tokens(off, l["entidades"]) for off, l in zip(enc.pop("offset_mapping"), laudos)]
    return Dataset.from_dict(dict(enc))


def metricas_entidades(p):
    logits, labels = p
    pred = np.argmax(logits, axis=-1)
    y_true, y_pred = [], []
    for ps, ls in zip(pred, labels):
        y_true.append([ID2LABEL[l] for l in ls if l != -100])
        y_pred.append([ID2LABEL[pp] for pp, l in zip(ps, ls) if l != -100])
    rel = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"f1": rel["micro avg"]["f1-score"], "precision": rel["micro avg"]["precision"],
            "recall": rel["micro avg"]["recall"]}


def avaliar_pipeline(extrator, laudos):
    total = [0, 0, 0]
    y_true, y_pred = [], []
    for l in laudos:
        dados = extrator.parse_texto(l["texto"])
        previsto = [(d["marcador"].id, d["valor"]) for d in dados]
        c = comparar_extracoes([(i["id"], i["valor"]) for i in l["itens"]], previsto)
        total = [a + b for a, b in zip(total, c)]
        risco = MotorDeRegras.avaliar_triagem_completa(dados).classificacao_geral.value if dados else "NORMAL"
        y_true.append(l["risco"])
        y_pred.append(risco)
    return {"extracao": prf(*total), "triagem": relatorio_classificacao(y_true, y_pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--saida", type=Path, default=RAIZ / "modelos" / "ner-bertimbau")
    ap.add_argument("--epocas", type=float, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=-1, help="para testes rápidos")
    ap.add_argument("--limite", type=int, default=None, help="usa só N laudos por split (teste rápido)")
    ap.add_argument("--retomar", action="store_true", help="continua do último checkpoint salvo em <saida>/checkpoints")
    args = ap.parse_args()

    dados = carregar(args.limite)
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    treino = tokenizar(dados["treino"], tokenizer, args.max_length)
    validacao = tokenizar(dados["validacao"], tokenizer, args.max_length)

    modelo = AutoModelForTokenClassification.from_pretrained(
        args.modelo, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    trainer = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir=str(args.saida / "checkpoints"),
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=10,
            learning_rate=args.lr, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epocas, max_steps=args.max_steps, weight_decay=0.01, warmup_steps=0.1,
            load_best_model_at_end=True, metric_for_best_model="f1", save_total_limit=1, report_to="none",
        ),
        train_dataset=treino,
        eval_dataset=validacao,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=metricas_entidades,
    )
    trainer.train(resume_from_checkpoint=True if args.retomar else None)
    trainer.save_model(str(args.saida))
    tokenizer.save_pretrained(str(args.saida))

    extrator = NerExtractor(str(args.saida), max_length=args.max_length)
    resultados = {"modelo_base": args.modelo, "historico": trainer.state.log_history}
    for split in ("teste", "teste_ood"):
        ds = tokenizar(dados[split], tokenizer, args.max_length)
        resultados[split] = {
            "entidades": trainer.evaluate(ds, metric_key_prefix=split),
            "pipeline_ner_regras": avaliar_pipeline(extrator, dados[split]),
        }

    saida = RAIZ / "resultados" / f"ner_{args.saida.name}.json"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for split in ("teste", "teste_ood"):
        e = resultados[split]["entidades"]
        p = resultados[split]["pipeline_ner_regras"]
        print(f"\n== {split}")
        print(f"Entidades (seqeval) P={e[f'{split}_precision']:.3f} R={e[f'{split}_recall']:.3f} F1={e[f'{split}_f1']:.3f}")
        print(f"Extração (marcador, valor) F1={p['extracao']['f1']:.3f}")
        t = p["triagem"]
        print(f"Triagem NER→regras acurácia={t['acuracia']:.3f} recall CRITICO={t['por_classe']['CRITICO']['recall']:.3f} "
              f"críticos perdidos={t['criticos_perdidos']}")
    print(f"\nModelo em {args.saida} | resultados em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
