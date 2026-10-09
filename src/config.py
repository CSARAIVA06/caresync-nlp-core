from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "CareSync NLP Core"
    GROQ_API_KEY: str = ""
    DEFAULT_LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TIMEOUT_SECONDS: float = 10.0
    # Extração: "regex" (padrão, leve) ou "ner" (BERTimbau treinado; requer requirements-ml.txt)
    EXTRATOR: str = "regex"
    NER_MODELO_DIR: str = "modelos/ner-bertimbau"
    # Geração: "auto" (Groq se houver chave, senão texto padrão), "groq", "ptt5" (local) ou "padrao"
    GERADOR: str = "auto"
    PTT5_MODELO_DIR: str = "modelos/ptt5-caresync"
    # Origens separadas por vírgula (ex: "https://caresync.vercel.app,http://localhost:3000")
    CORS_ORIGINS: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
