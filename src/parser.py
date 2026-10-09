import re
from typing import Any, Dict, List

# Número com separadores opcionais: "13,5", "7.500", "250.000", "1.250,5"
_NUM = r"(\d+(?:[.,]\d+)*)"


def _build_pattern(nome: str, unidades: str) -> re.Pattern:
    # Sem DOTALL e sem \s: valor e referência precisam estar na mesma linha do marcador.
    return re.compile(
        rf"{nome}[: \t]+{_NUM}[ \t]*({unidades})?.*?refer[êe]ncia[: \t]+{_NUM}[ \t]*(?:a|-|até)[ \t]*{_NUM}",
        re.IGNORECASE,
    )


def parse_numero(texto: str) -> float:
    """Converte números no formato brasileiro para float.

    "250.000" -> 250000.0 (ponto como separador de milhar)
    "13,5"    -> 13.5
    "1.250,5" -> 1250.5
    "13.5"    -> 13.5
    """
    if "," in texto:
        return float(texto.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", texto):
        return float(texto.replace(".", ""))
    return float(texto)


class LabParser:
    """Extrai informações numéricas e textuais de laudos laboratoriais brutos."""

    PATTERNS = {
        "Hemoglobina": _build_pattern(r"hemoglobina", r"g/dL"),
        "Leucocitos": _build_pattern(r"leuc[óo]citos", r"/mm³|/mm3|/µL|/uL"),
        "Plaquetas": _build_pattern(r"plaquetas", r"/mm³|/mm3|/µL|/uL"),
        "Ferritina": _build_pattern(r"ferritina", r"ng/mL"),
    }

    @classmethod
    def parse_texto(cls, texto: str) -> List[Dict[str, Any]]:
        resultados = []
        for marcador, pattern in cls.PATTERNS.items():
            match = pattern.search(texto)
            if not match:
                continue
            valor, unidade, ref_min, ref_max = match.groups()
            resultados.append({
                "parametro": marcador,
                "valor": parse_numero(valor),
                "unidade": unidade or "",
                "ref_min": parse_numero(ref_min),
                "ref_max": parse_numero(ref_max),
            })
        return resultados
