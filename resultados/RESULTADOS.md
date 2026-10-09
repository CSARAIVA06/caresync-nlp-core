# Resultados experimentais — CareSync

Dataset sintético (seed 42): 630 laudos de treino, 135 de validação, 135 de teste (formatos vistos) e 300 de teste OOD (formatos nunca vistos no treino).

## Extração e triagem de risco

| Sistema | Split | F1 extração | Acurácia risco | F1-macro risco | Recall CRÍTICO | Críticos perdidos |
|---|---|---|---|---|---|---|
| Regex + regras | teste | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| NER (BERTimbau) + regras | teste | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| TF-IDF + SVM | teste | — | 0.467 | 0.467 | 0.500 | 20 |
| Híbrido (SVM + regras/regex) | teste | — | 0.615 | 0.602 | 1.000 | 0 |
| Regex + regras | teste_ood | 0.014 | 0.350 | 0.201 | 0.000 | 100 |
| NER (BERTimbau) + regras | teste_ood | 0.855 | 0.893 | 0.893 | 0.860 | 14 |
| TF-IDF + SVM | teste_ood | — | 0.427 | 0.419 | 0.450 | 55 |
| Híbrido (SVM + regras/regex) | teste_ood | — | 0.427 | 0.419 | 0.450 | 55 |

## NER — entidades (seqeval, micro)

| Split | Precision | Recall | F1 |
|---|---|---|---|
| teste | 0.999 | 1.000 | 1.000 |
| teste_ood | 0.888 | 0.951 | 0.918 |

_Rótulos de risco gerados pelas mesmas regras determinísticas da API; por isso o motor de regras é exato quando a extração é correta. Ver seção de limitações._
