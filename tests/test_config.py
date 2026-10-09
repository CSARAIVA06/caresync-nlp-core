import pytest
from pydantic import ValidationError

from src.config import Settings


def _settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_aceita_maiusculas_e_espacos():
    s = _settings(GERADOR=" PTT5 ", EXTRATOR="NER")
    assert s.GERADOR == "ptt5" and s.EXTRATOR == "ner"


@pytest.mark.parametrize("campo, valor", [("GERADOR", "llm"), ("EXTRATOR", "bert")])
def test_valor_invalido_impede_iniciar(campo, valor):
    with pytest.raises(ValidationError):
        _settings(**{campo: valor})


def test_groq_sem_chave_impede_iniciar():
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        _settings(GERADOR="groq", GROQ_API_KEY="")
