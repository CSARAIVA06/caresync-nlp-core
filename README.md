# CareSync NLP Core

- **Site e simulador no ar:** https://caresync-eta-ten.vercel.app
- **Código do site:** [CSARAIVA06/caresync](https://github.com/CSARAIVA06/caresync)
- **Resultados dos experimentos:** [`resultados/RESULTADOS.md`](resultados/RESULTADOS.md)
- **Artigo:** [CareSync — PLN e RAG (Google Docs)](https://docs.google.com/document/d/19oNTGBU-E0Cj5yOpQhvgTSQfEZarA5vXmbHH8mtzhHg/view)

Trabalho da disciplina de Processamento de Linguagem Natural — Ciência de Dados, Fatec Jundiaí.
Autor: Caio Roberto Farias Saraiva · Orientação: Leonardo de Lellis Rossi.

API de triagem de laudos laboratoriais do [CareSync](https://caresync-eta-ten.vercel.app). Recebe o texto bruto de um laudo e devolve:

1. os marcadores extraídos (valor, unidade e faixa de referência);
2. a classificação de risco (`NORMAL`, `ATENCAO`, `CRITICO`), feita por regras determinísticas;
3. uma mensagem humanizada para o paciente, gerada por LLM (Groq), com guardrails.

O site ([CSARAIVA06/caresync](https://github.com/CSARAIVA06/caresync)) chama esta API no simulador e usa o motor do navegador como reserva caso ela esteja fora do ar.

## Arquitetura

```
texto do laudo
  └─► parser.py         NER por regras: marcadores, valores, referência do laudo
       └─► rules.py     classificação determinística (pior item define o risco)
            └─► llm_service.py   LLM escreve a explicação + verificação de guardrails
                 └─► mensagens.py    saudação, conduta e aviso fixos
```

| Módulo | Função |
|---|---|
| `src/knowledge_base.py` | Base de conhecimento (12 marcadores): aliases, unidade, referência, limites críticos e explicações em linguagem simples. É o contexto do "RAG". Mantida em sincronia com `app/lib/triage.js` do site. |
| `src/parser.py` | Extrai os marcadores de texto livre. Entende números brasileiros (`13,5`, `250.000`, `250 mil`), ignora datas e "hemoglobina glicada"/"colesterol HDL", e lê a referência do laudo (`VR: 12,0 a 15,5`, `VR: < 190`). Se o laudo não traz referência, ou ela está em outra escala, usa a da base. |
| `src/rules.py` | Classifica cada item: fora da referência → `ATENCAO`; além do limite crítico da base → `CRITICO`. Os limites críticos nunca vêm do laudo. |
| `src/llm_service.py` | O LLM escreve só a explicação dos resultados. A resposta é descartada se afirmar diagnóstico ou prescrever algo, e o texto padrão é usado. O nome do paciente **nunca** é enviado ao LLM. |
| `src/mensagens.py` | Texto padrão e partes fixas da mensagem (saudação, orientação de conduta, como "ligue 192", e aviso). |
| `src/main.py` | API FastAPI. |

## Pesquisa: modelos treinados

Além do pipeline de regras, o repositório tem os experimentos de PLN do artigo. Todos usam
um dataset sintético reprodutível e são comparados com a linha de base de regex.

| Fase | O que é | Onde |
|---|---|---|
| 1. Dados | 1.200 laudos sintéticos em 8 formatos, com entidades anotadas (posições exatas), rótulo de risco e pares entrada→explicação. Dois formatos ficam só no teste (**OOD**) para medir a generalização. Exporta `ner_bio.jsonl`, `classificacao.csv` e `seq2seq.jsonl`. | `scripts/gerar_dataset.py` → `data/` |
| Linha de base | Regex + regras nos mesmos splits. | `scripts/avaliar_regex.py` |
| 2. NER | Fine-tuning do **BERTimbau** (Token Classification, BIO: EXAME, VALOR, UNIDADE, REFERENCIA). `src/ner.py` substitui o regex sem mudar as regras. O notebook mostra a curva de loss, a comparação com o regex e as matrizes de confusão (também em `resultados/*.png`). | `scripts/treinar_ner.py`, `notebooks/02_ner_bertimbau.ipynb` |
| 3. Classificador | **TF-IDF + SVM** prevendo o risco direto do texto, comparado com as regras e com a versão **híbrida** (regras sobrescrevem para CRÍTICO). | `scripts/treinar_classificador.py` |
| 4. Geração | Fine-tuning do **PTT5** (encoder-decoder): resultado estruturado → explicação em linguagem simples. **Implementado, mas não executado nesta versão** (sem modelo nem resultados; ver Limitações do artigo, H3). | `scripts/treinar_ptt5.py`, `notebooks/03_fine_tuning_ptt5.ipynb` |
| 5. Inferência local | A API carrega os modelos treinados: `EXTRATOR=ner` (e `GERADOR=ptt5`, quando o PTT5 for treinado). Com o NER, a extração não sai do servidor. | `src/ner.py`, `src/gerador_local.py` |
| 6. Métricas | P/R/F1, matriz de confusão e críticos perdidos (reportados). ROUGE-L, Flesch PT-BR e fidelidade numérica estão implementados para a avaliação do PTT5. | `src/metricas.py` → `resultados/RESULTADOS.md` |

### Reproduzindo

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu   # no Colab o torch já vem instalado
pip install -r requirements-ml.txt

python scripts/gerar_dataset.py          # Fase 1
python scripts/avaliar_regex.py          # linha de base
python scripts/treinar_classificador.py  # Fase 3 (segundos, CPU)
python scripts/treinar_ner.py            # Fase 2 (~26 min na CPU; minutos numa GPU T4)
python scripts/treinar_ptt5.py           # Fase 4, não executada no artigo (use GPU: Colab T4)
python scripts/compilar_resultados.py    # tabela para o artigo
```

Para rodar a API com os modelos treinados (fora do Vercel, que não comporta PyTorch):

```powershell
$env:EXTRATOR="ner"; $env:GERADOR="ptt5"; uvicorn src.main:app
```

### Decisões de projeto e limitações

- **O modelo nunca decide o risco.** O NER só *lê* o laudo; a comparação com os limites é feita pelas regras. O PTT5 só escreve a explicação; a conduta (ex.: ligar 192) é fixa.
- **Guardrails pós-geração:** textos com diagnóstico/prescrição ou com números que não estão na entrada são descartados e substituídos pelo texto padrão.
- **Rótulos sintéticos:** os níveis de risco do dataset são calculados pelas próprias regras, então as regras são exatas *por construção* quando a extração é correta. As explicações-alvo vêm de modelos de frase; o campo `explicacao_revisada` existe para revisão humana (idealmente por profissional de saúde) antes de usar ROUGE-L como medida de qualidade.
- **Formatos vistos x OOD:** o regex foi escrito para os formatos "vistos"; o desempenho real de generalização aparece no split `teste_ood`.
- **Erros conhecidos do NER (teste OOD):** no formato "tabela com a referência antes do resultado" (F1 de extração 0,933), o modelo pegou o limite da referência como VALOR em 37 marcadores e deixou de extrair outros 6. Ele aprendeu que "o número após o nome do exame é o resultado". No laudo em várias linhas (F1 0,982), o erro é outro: 21 marcadores não extraídos. Esses erros causam 6 dos 7 críticos perdidos (1 no teste, formato sem referência); 6 dos 7 viraram NORMAL. Incluir tabelas com colunas em ordens variadas e laudos quebrados em linhas no treino deve reduzir isso.
- **Reprodutibilidade:** `data/` não é versionado; `python scripts/gerar_dataset.py --seed 42` recria exatamente o mesmo dataset. Resultados em `resultados/RESULTADOS.md`.
- **Modelos não versionados:** `modelos/` (o NER treinado tem 433 MB) fica fora do git. Para usar `EXTRATOR=ner` ou reproduzir a Fase 2, rode `scripts/treinar_ner.py`. Os testes, a API (com regex), o dataset, a linha de base e o classificador rodam sem ele.
- **Configuração do NER publicado:** `neuralmind/bert-base-portuguese-cased`, 4 épocas (316 passos), batch 8, taxa de aprendizado 5×10⁻⁵ com aquecimento linear em 10% dos passos, weight decay 0,01, máximo de 512 tokens, melhor época escolhida pelo F1 na validação. Treino de 1.532,6 s (~25,5 min) em CPU (AMD Ryzen 5 5500, 32 GB de RAM, PyTorch sem GPU); inferência a ~8 laudos/s.

## Rodando localmente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                # preencha GROQ_API_KEY (opcional)

uvicorn src.main:app --reload
```

Documentação interativa: http://localhost:8000/docs

## Testes

```bash
pytest
```

Incluem os laudos de exemplo do site, para garantir que a API e o motor do navegador chegam ao mesmo nível de risco.

## Deploy no Vercel

1. No Vercel: **Add New → Project** e importe este repositório do GitLab. O FastAPI é detectado sozinho (`app.py` na raiz).
2. Em **Settings → Environment Variables** adicione:
   - `GROQ_API_KEY`: chave criada em https://console.groq.com/keys
   - `CORS_ORIGINS`: `https://caresync-eta-ten.vercel.app,http://localhost:3000`
3. Faça o deploy e teste `https://<seu-projeto>.vercel.app/docs`.
4. No projeto do **site**, defina `NEXT_PUBLIC_TRIAGE_API_URL=https://<seu-projeto>.vercel.app` e faça um novo deploy.

Também há um `Dockerfile` para outras plataformas (Render, Railway, Fly): `docker build -t caresync-nlp-core .`

## API

### `POST /api/v1/triagem`

```json
{
  "paciente_id": "P-123",
  "paciente_nome": "José Almeida",
  "texto_laudo": "HEMOGRAMA\nHemoglobina: 6,4 g/dL (VR: 12,0 a 17,5)"
}
```

`paciente_id` e `paciente_nome` são opcionais; o nome só é usado na saudação. O laudo pode ter até 10.000 caracteres.

Resposta (resumida):

```json
{
  "paciente_id": "P-123",
  "tipo_exame": "Hemograma",
  "classificacao_geral": "CRITICO",
  "cor_badge": "#EF4444",
  "exames_extraidos": [
    {
      "id": "hemoglobina", "parametro": "Hemoglobina", "valor": 6.4, "unidade": "g/dL",
      "ref_min": 12.0, "ref_max": 17.5, "ref_fonte": "laudo",
      "status_item": "CRITICO", "direcao": "baixo"
    }
  ],
  "mensagem_humanizada": "Olá, José!\n...",
  "fonte_mensagem": "llm",
  "fontes": ["Hemoglobina: proteína do sangue que transporta oxigênio pelo corpo. Referência (adulto): 12 a 17,5 g/dL."],
  "guardrails": ["Nenhum diagnóstico é afirmado, apenas a explicação do resultado.", "..."]
}
```

Retorna `400` quando nenhum marcador é reconhecido e `422` para entrada inválida.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `GROQ_API_KEY` | vazio | Chave da API Groq. Sem ela, a API usa o texto padrão. |
| `DEFAULT_LLM_MODEL` | `llama-3.3-70b-versatile` | Modelo usado no Groq. |
| `LLM_TIMEOUT_SECONDS` | `10` | Tempo máximo de espera pelo LLM antes de usar o texto padrão. |
| `CORS_ORIGINS` | `*` | Origens permitidas, separadas por vírgula. |
| `EXTRATOR` | `regex` | `regex` ou `ner` (modelo em `NER_MODELO_DIR`, padrão `modelos/ner-bertimbau`). |
| `GERADOR` | `auto` | `auto` (Groq se houver chave), `groq`, `ptt5` (modelo em `PTT5_MODELO_DIR`, padrão `modelos/ptt5-caresync`) ou `padrao`. |

> ⚠️ Projeto acadêmico de demonstração. Os intervalos são referências gerais para adultos e o serviço não substitui avaliação médica.
