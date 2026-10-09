# CareSync NLP Core

API de triagem de laudos laboratoriais do [CareSync](https://caresync-eta-ten.vercel.app). Recebe o texto bruto de um laudo e devolve:

1. os marcadores extraídos (valor, unidade e faixa de referência);
2. a classificação de risco (`NORMAL`, `ATENCAO`, `CRITICO`), feita por regras determinísticas;
3. uma mensagem humanizada para o paciente, gerada por LLM (Groq), com guardrails.

O site ([gitlab.com/caiofasaraiva/caresync](https://gitlab.com/caiofasaraiva/caresync)) chama esta API no simulador e usa o motor do navegador como reserva caso ela esteja fora do ar.

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

> ⚠️ Projeto acadêmico de demonstração. Os intervalos são referências gerais para adultos e o serviço não substitui avaliação médica.
