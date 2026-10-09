"""Base de conhecimento dos marcadores laboratoriais (fonte do "RAG").

Mantida em sincronia com `MARKERS` em app/lib/triage.js do site CareSync.
Os intervalos são valores de referência gerais para adultos, apenas para demonstração.
"""
import re
from dataclasses import dataclass, field
from typing import Optional, Pattern, Tuple

Faixa = Tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class Marcador:
    id: str
    nome: str
    aliases: Tuple[str, ...]
    unidade: str
    ref: Faixa
    critico: Faixa
    explicacao: str
    baixo: str
    alto: str
    # Termos que, logo após o alias, indicam outro exame (ex.: "hemoglobina glicada").
    exclude: Optional[Pattern] = field(default=None, compare=False)


MARCADORES = (
    Marcador(
        id="hemoglobina", nome="Hemoglobina", aliases=("hemoglobina", "hb", "hgb"),
        exclude=re.compile(r"^\s*(glicada|glicosilada|a1c)"),
        unidade="g/dL", ref=(12, 17.5), critico=(7, 20),
        explicacao="proteína do sangue que transporta oxigênio pelo corpo",
        baixo="pode indicar anemia, que causa cansaço e fraqueza",
        alto="pode estar relacionado a desidratação ou outras condições",
    ),
    Marcador(
        id="hematocrito", nome="Hematócrito", aliases=("hematocrito", "ht", "hct"),
        unidade="%", ref=(36, 52), critico=(21, 60),
        explicacao="porcentagem do sangue ocupada pelas células vermelhas",
        baixo="costuma acompanhar quadros de anemia",
        alto="pode indicar desidratação ou excesso de células vermelhas",
    ),
    Marcador(
        id="leucocitos", nome="Leucócitos", aliases=("leucocitos", "globulos brancos", "wbc"),
        unidade="/mm³", ref=(4000, 11000), critico=(2000, 30000),
        explicacao="contagem das células de defesa do organismo",
        baixo="pode reduzir a capacidade do corpo de combater infecções",
        alto="costuma aparecer quando o corpo está combatendo uma infecção ou inflamação",
    ),
    Marcador(
        id="plaquetas", nome="Plaquetas", aliases=("plaquetas", "plt"),
        unidade="/mm³", ref=(150000, 450000), critico=(20000, 1000000),
        explicacao="contagem das células que ajudam o sangue a coagular",
        baixo="pode aumentar o risco de sangramentos e manchas roxas",
        alto="pode aumentar o risco de formação de coágulos",
    ),
    Marcador(
        id="glicose", nome="Glicose (jejum)", aliases=("glicose", "glicemia"),
        unidade="mg/dL", ref=(70, 99), critico=(50, 400),
        explicacao="quantidade de açúcar no sangue",
        baixo="pode causar tontura, suor e confusão",
        alto="merece avaliação para pré-diabetes ou diabetes",
    ),
    Marcador(
        id="potassio", nome="Potássio", aliases=("potassio", "k+"),
        unidade="mEq/L", ref=(3.5, 5.1), critico=(2.8, 6.2),
        explicacao="dosagem de um mineral importante para os músculos e o coração",
        baixo="pode causar cãibras, fraqueza e alterações no ritmo do coração",
        alto="pode alterar o ritmo do coração",
    ),
    Marcador(
        id="sodio", nome="Sódio", aliases=("sodio", "na+"),
        unidade="mEq/L", ref=(135, 145), critico=(120, 160),
        explicacao="dosagem de um mineral que regula a água no corpo",
        baixo="pode causar enjoo, dor de cabeça e confusão",
        alto="costuma estar ligado a desidratação",
    ),
    Marcador(
        id="creatinina", nome="Creatinina", aliases=("creatinina",),
        unidade="mg/dL", ref=(0.6, 1.3), critico=(None, 5),
        explicacao="dosagem da substância que mostra como os rins estão filtrando o sangue",
        baixo="geralmente não é motivo de preocupação",
        alto="pode indicar que os rins estão filtrando menos do que o esperado",
    ),
    Marcador(
        id="ureia", nome="Ureia", aliases=("ureia",),
        unidade="mg/dL", ref=(15, 45), critico=(None, 200),
        explicacao="dosagem de outro marcador do funcionamento dos rins",
        baixo="geralmente não é motivo de preocupação",
        alto="pode indicar desidratação ou alteração na função dos rins",
    ),
    Marcador(
        id="colesterol", nome="Colesterol total", aliases=("colesterol total", "colesterol"),
        exclude=re.compile(r"^\s*(hdl|ldl|vldl|nao)"),
        unidade="mg/dL", ref=(None, 190), critico=(None, None),
        explicacao="medida da gordura do sangue ligada à saúde do coração",
        baixo="",
        alto="aumenta o risco cardiovascular a longo prazo",
    ),
    Marcador(
        id="triglicerideos", nome="Triglicerídeos", aliases=("triglicerideos", "triglicerides"),
        unidade="mg/dL", ref=(None, 150), critico=(None, 1000),
        explicacao="medida de outro tipo de gordura do sangue, ligada à alimentação",
        baixo="",
        alto="se relaciona à alimentação e ao risco cardiovascular",
    ),
    Marcador(
        id="tsh", nome="TSH", aliases=("tsh",),
        unidade="µUI/mL", ref=(0.4, 4.5), critico=(None, None),
        explicacao="dosagem do hormônio que controla o funcionamento da tireoide",
        baixo="pode indicar tireoide mais acelerada que o normal",
        alto="pode indicar tireoide mais lenta que o normal",
    ),
)

POR_ID = {m.id: m for m in MARCADORES}


def formatar_numero(v: float) -> str:
    """Formata no padrão pt-BR com até 2 casas: 245000 -> "245.000", 6.4 -> "6,4"."""
    texto = f"{v:,.2f}".rstrip("0").rstrip(".")
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def formatar_faixa(ref: Faixa, unidade: str) -> str:
    lo, hi = ref
    if lo is None:
        return f"até {formatar_numero(hi)} {unidade}"
    if hi is None:
        return f"acima de {formatar_numero(lo)} {unidade}"
    return f"{formatar_numero(lo)} a {formatar_numero(hi)} {unidade}"
