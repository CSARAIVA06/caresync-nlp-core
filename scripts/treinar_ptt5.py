"""Fase 4: fine-tuning do PTT5 (encoder-decoder) para explicar exames em linguagem simples.

Entrada: "triagem: CRITICO | Hemoglobina: 6,4 g/dL (ref 12 a 17,5 g/dL; baixo) | ..."
Saída:   explicação ao paciente (sem saudação/conduta/aviso, que são fixos na API).

Correções em relação ao exemplo do guia:
- validação em split separado (o exemplo avaliava no próprio treino);
- DataCollatorForSeq2Seq (sem ele o treino quebra com rótulos de tamanhos diferentes);
- learning rate 3e-4 (2e-5 é baixo demais para T5) e limites de tokens maiores
  (128 truncava as explicações);
- `eval_strategy` e `processing_class` (nomes atuais da biblioteca).

Avaliação no split "teste": ROUGE-L, Flesch PT-BR (laudo bruto x texto gerado),
fidelidade numérica (nenhum número inventado) e violações de guardrail.

Uso (GPU no Colab):
    python scripts/treinar_ptt5.py
Teste rápido na CPU:
    python scripts/treinar_ptt5.py --modelo unicamp-dl/ptt5-small-portuguese-vocab --limite 24 --max-steps 4 --saida modelos/ptt5-teste
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.gerador_local import GeradorPTT5  # noqa: E402
from src.llm_service import viola_guardrails  # noqa: E402
from src.metricas import fidelidade_numerica, flesch_ptbr, rouge_l  # noqa: E402


def carregar(limite=None):
    linhas = [json.loads(l) for l in open(RAIZ / "data" / "seq2seq.jsonl", encoding="utf-8")]
    laudos = {}
    for linha in open(RAIZ / "data" / "laudos_sinteticos.jsonl", encoding="utf-8"):
        laudo = json.loads(linha)
        laudos[laudo["id"]] = laudo["texto"]
    por_split = defaultdict(list)
    for l in linhas:
        l["laudo_bruto"] = laudos[l["id"]]
        por_split[l["split"]].append(l)
    if limite:
        por_split = {s: v[:limite] for s, v in por_split.items()}
    return por_split


def tokenizar(linhas, tokenizer, max_input, max_target):
    enc = tokenizer([l["input_text"] for l in linhas], max_length=max_input, truncation=True)
    enc["labels"] = tokenizer(text_target=[l["target_text"] for l in linhas], max_length=max_target,
                              truncation=True)["input_ids"]
    return Dataset.from_dict(dict(enc))


def avaliar_geracao(gerador, linhas, batch=8):
    gerados = []
    for i in range(0, len(linhas), batch):
        gerados += gerador.gerar_textos([l["input_text"] for l in linhas[i:i + batch]])
    amostras = []
    for l, g in zip(linhas, gerados):
        amostras.append({
            "id": l["id"], "risco": l["risco"], "entrada": l["input_text"], "referencia": l["target_text"], "gerado": g,
            "rouge_l": rouge_l(l["target_text"], g),
            "flesch_laudo": flesch_ptbr(l["laudo_bruto"]), "flesch_gerado": flesch_ptbr(g),
            "fiel_numeros": fidelidade_numerica(l["input_text"], g),
            "viola_guardrail": viola_guardrails(g),
        })
    media = lambda k: statistics.mean(a[k] for a in amostras)  # noqa: E731
    resumo = {
        "rouge_l": media("rouge_l"),
        "flesch_laudo_bruto": media("flesch_laudo"),
        "flesch_gerado": media("flesch_gerado"),
        "flesch_referencia": statistics.mean(flesch_ptbr(l["target_text"]) for l in linhas),
        "fidelidade_numerica": media("fiel_numeros"),
        "taxa_violacao_guardrail": media("viola_guardrail"),
    }
    return resumo, amostras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="unicamp-dl/ptt5-base-portuguese-vocab")
    ap.add_argument("--saida", type=Path, default=RAIZ / "modelos" / "ptt5-caresync")
    ap.add_argument("--epocas", type=float, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-input", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=384)
    ap.add_argument("--max-steps", type=int, default=-1, help="para testes rápidos")
    ap.add_argument("--limite", type=int, default=None, help="usa só N exemplos por split (teste rápido)")
    ap.add_argument("--retomar", action="store_true", help="continua do último checkpoint salvo em <saida>/checkpoints")
    args = ap.parse_args()

    dados = carregar(args.limite)
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(args.modelo)
    treino = tokenizar(dados["treino"], tokenizer, args.max_input, args.max_target)
    validacao = tokenizar(dados["validacao"], tokenizer, args.max_input, args.max_target)

    trainer = Seq2SeqTrainer(
        model=modelo,
        args=Seq2SeqTrainingArguments(
            output_dir=str(args.saida / "checkpoints"),
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=10,
            learning_rate=args.lr, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epocas, max_steps=args.max_steps, weight_decay=0.01, warmup_steps=0.05,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            save_total_limit=1, report_to="none",
        ),
        train_dataset=treino,
        eval_dataset=validacao,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=modelo),
    )
    trainer.train(resume_from_checkpoint=True if args.retomar else None)
    trainer.save_model(str(args.saida))
    tokenizer.save_pretrained(str(args.saida))

    gerador = GeradorPTT5(str(args.saida), max_input=args.max_input, max_new_tokens=args.max_target)
    resumo, amostras = avaliar_geracao(gerador, dados["teste"])

    saida = RAIZ / "resultados" / f"ptt5_{args.saida.name}.json"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps({"modelo_base": args.modelo, "teste": resumo, "historico": trainer.state.log_history,
                                 "amostras": amostras}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== teste")
    for k, v in resumo.items():
        print(f"{k:24s} {v:.3f}")
    print("\nExemplo:")
    print("ENTRADA:", amostras[0]["entrada"])
    print("GERADO :", amostras[0]["gerado"])
    print(f"\nModelo em {args.saida} | resultados em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
