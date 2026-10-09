"""Partes determinísticas da mensagem ao paciente (mesmos textos de buildMessage no site).

A mensagem final é: saudação + corpo + orientação de conduta + aviso.
O corpo pode vir do LLM; as demais partes são sempre fixas, para que a orientação
de segurança (ex.: procurar atendimento / SAMU) nunca dependa do modelo.
"""
from typing import Optional

from src.knowledge_base import POR_ID, formatar_faixa, formatar_numero
from src.schemas import ItemExame, NivelRisco, ResultadoTriagem

INTRO = {
    NivelRisco.NORMAL: "Seus resultados chegaram e temos uma boa notícia: todos os exames analisados estão dentro dos valores esperados.",
    NivelRisco.ATENCAO: "Seus resultados chegaram. A maior parte está dentro do esperado, mas alguns valores merecem a atenção do seu médico. Isso é comum e não indica uma emergência.",
    NivelRisco.CRITICO: "Seus resultados chegaram e um ou mais valores precisam de avaliação médica rápida. Nossa equipe de saúde já foi avisada e vai entrar em contato com você.",
}

CONDUTA = {
    NivelRisco.NORMAL: "Não é necessária nenhuma ação imediata. Leve estes resultados na sua próxima consulta de rotina.",
    NivelRisco.ATENCAO: "Recomendamos agendar uma consulta de retorno nos próximos dias para avaliar esses resultados com calma. Você pode marcar pelo nosso canal de atendimento.",
    NivelRisco.CRITICO: "Por favor, procure atendimento ainda hoje. Se sentir falta de ar, dor no peito, desmaio, sangramento ou fraqueza intensa, vá imediatamente ao pronto-socorro ou ligue 192 (SAMU).",
}

AVISO = "Esta mensagem explica seus resultados em linguagem simples e não substitui a avaliação do seu médico."


def faixa_item(item: ItemExame) -> str:
    return formatar_faixa((item.ref_min, item.ref_max), item.unidade)


def _explicar_alterado(item: ItemExame) -> str:
    m = POR_ID[item.id]
    lado = "abaixo" if item.direcao == "baixo" else "acima"
    consequencia = m.baixo if item.direcao == "baixo" else m.alto
    return (
        f"• {item.parametro}: {formatar_numero(item.valor)} {item.unidade} ({lado} da referência de "
        f"{faixa_item(item)}). É a {m.explicacao}. Esse resultado {consequencia}."
    )


def corpo_padrao(triagem: ResultadoTriagem) -> str:
    nivel = triagem.classificacao_geral
    if nivel == NivelRisco.NORMAL:
        linhas = [
            f"• {i.parametro}: {formatar_numero(i.valor)} {i.unidade}. É a {POR_ID[i.id].explicacao}, e está normal."
            for i in triagem.exames_extraidos
        ]
    else:
        linhas = [_explicar_alterado(i) for i in triagem.exames_extraidos if i.status_item != NivelRisco.NORMAL]
    return "\n".join([INTRO[nivel], "", *linhas])


def saudacao(nome: Optional[str]) -> str:
    primeiro = nome.strip().split()[0] if nome and nome.strip() else ""
    return f"Olá, {primeiro}!" if primeiro else "Olá!"


def montar_mensagem(triagem: ResultadoTriagem, corpo: str, nome: Optional[str]) -> str:
    return "\n".join([saudacao(nome), corpo.strip(), "", CONDUTA[triagem.classificacao_geral], "", AVISO])
