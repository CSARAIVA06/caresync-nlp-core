# Resultados experimentais — CareSync

Dataset sintético (seed 42): 630 laudos de treino, 135 de validação, 135 de teste (formatos vistos) e 300 de teste OOD (formatos nunca vistos no treino).

## Extração e triagem de risco

| Sistema | Split | F1 extração | Acurácia risco | F1-macro risco | Recall CRÍTICO | Críticos perdidos |
|---|---|---|---|---|---|---|
| Regex + regras | teste | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| NER (BERTimbau) + regras | teste | 0.999 | 0.993 | 0.992 | 0.979 | 1 |
| TF-IDF + SVM | teste | — | 0.393 | 0.389 | 0.489 | 24 |
| Híbrido (SVM + regras/regex) | teste | — | 0.570 | 0.528 | 1.000 | 0 |
| Regex + regras | teste_ood | 0.008 | 0.323 | 0.180 | 0.000 | 99 |
| NER (BERTimbau) + regras | teste_ood | 0.957 | 0.960 | 0.960 | 0.939 | 6 |
| TF-IDF + SVM | teste_ood | — | 0.490 | 0.490 | 0.515 | 48 |
| Híbrido (SVM + regras/regex) | teste_ood | — | 0.490 | 0.490 | 0.515 | 48 |

## NER — entidades (seqeval, micro)

| Split | Precision | Recall | F1 |
|---|---|---|---|
| teste | 0.999 | 0.999 | 0.999 |
| teste_ood | 0.977 | 0.984 | 0.980 |

_Rótulos de risco gerados pelas mesmas regras determinísticas da API; por isso o motor de regras é exato quando a extração é correta. Ver seção de limitações._
