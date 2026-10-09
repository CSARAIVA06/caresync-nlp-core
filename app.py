# Ponto de entrada para o Vercel (detecta `app` em app.py na raiz).
# Localmente e no Docker use: uvicorn src.main:app
from src.main import app  # noqa: F401
