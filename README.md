# CareSync NLP Core

Serviço de triagem de laudos laboratoriais do CareSync. Recebe o texto bruto de um laudo, extrai os marcadores (Hemoglobina, Leucócitos, Plaquetas, Ferritina), classifica o risco de forma determinística (`NORMAL`, `ATENCAO`, `CRITICO`) e gera uma mensagem humanizada para o paciente via LLM (Groq), com guardrails éticos.

## Arquitetura

```
texto do laudo ──► parser.py (regex/NER) ──► rules.py (classificação determinística) ──► llm_service.py (mensagem) ──► API
```

- **`src/parser.py`**: extrai valor, unidade e faixa de referência. Entende números no formato brasileiro (`13,5`, `250.000`).
- **`src/rules.py`**: classifica cada item e o resultado geral (o pior item define o risco). Valores de pânico: hemoglobina < 7,0 g/dL e plaquetas < 50.000/mm³ → `CRITICO`.
- **`src/llm_service.py`**: gera a mensagem para o paciente. Sem `GROQ_API_KEY`, ou se o LLM falhar, usa uma mensagem fixa por nível de risco. A classificação nunca depende do LLM.
- **`src/main.py`**: API FastAPI.

## Rodando localmente

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # preencha GROQ_API_KEY (opcional)

uvicorn src.main:app --reload
```

Documentação interativa: http://localhost:8000/docs

## Testes

```bash
pytest
```

## Docker

```bash
docker build -t caresync-nlp-core .
docker run -p 8000:8000 --env-file .env caresync-nlp-core
```

A porta pode ser alterada com a variável `PORT` (útil para Render/Railway/Fly).

## API

### `POST /api/v1/triagem`

```json
{
  "paciente_id": "P-123",
  "texto_laudo": "Hemoglobina: 6,2 g/dL Referência: 12,0 a 15,5"
}
```

Resposta:

```json
{
  "paciente_id": "P-123",
  "tipo_exame": "Análise de Sangue / Hemograma",
  "classificacao_geral": "CRITICO",
  "cor_badge": "#EF4444",
  "exames_extraidos": [
    {"parametro": "Hemoglobina", "valor": 6.2, "unidade": "g/dL", "ref_min": 12.0, "ref_max": 15.5, "status_item": "CRITICO"}
  ],
  "mensagem_humanizada": "..."
}
```

Retorna `400` quando nenhum marcador é reconhecido no texto.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `GROQ_API_KEY` | vazio | Chave da API Groq. Sem ela, usa mensagens padrão. |
| `DEFAULT_LLM_MODEL` | `llama-3.3-70b-versatile` | Modelo usado no Groq. |
| `CORS_ORIGINS` | `*` | Origens permitidas, separadas por vírgula (ex: o domínio do Vercel). |

> ⚠️ Este serviço é um apoio à comunicação e não substitui avaliação médica.
