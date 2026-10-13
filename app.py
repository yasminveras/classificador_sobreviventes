import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="SpaceTec-Info EAJ | Previsão de Resgate",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 2. CARREGAMENTO DO MODELO
# ==========================================

CAMINHO_MODELO = Path(__file__).parent / "modelo_final.pkl"


@st.cache_resource
def carregar_modelo(caminho):
    with open(caminho, "rb") as arquivo:
        return pickle.load(arquivo)


try:
    if not CAMINHO_MODELO.exists():
        st.error(
            "O arquivo modelo_final.pkl não foi encontrado. "
            "Verifique se ele está na mesma pasta do app.py."
        )
        st.stop()

    pipeline = carregar_modelo(str(CAMINHO_MODELO))

except Exception as erro:
    st.error("Não foi possível carregar o modelo de previsão.")
    st.caption(f"Detalhes técnicos: {erro}")
    st.stop()


# ==========================================
# 3. CABEÇALHO DO APLICATIVO
# ==========================================

st.title("🛰️ SpaceTec-Info EAJ")
st.subheader("Previsão de resgate de tripulantes")

st.markdown(
    """
    Simule cenários de uma emergência espacial e utilize o modelo de
    aprendizado de máquina para estimar a probabilidade de resgate
    de um tripulante após o impacto de micrometeorito na estação.
    """
)

st.divider()


# ==========================================
# 4. FORMULÁRIO DE ENTRADA
# ==========================================

st.header("📋 Dados do tripulante")

st.write(
    "Preencha as informações abaixo e clique em "
    "**Prever resgate** para consultar o resultado."
)

with st.form("formulario_tripulante"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚀 Informações da estação")

        modulo_estacao = st.selectbox(
            "Módulo da estação",
            ["Comando", "Laboratorio", "Habitacao", "Engenharia"]
        )

        classe_bilhete = st.selectbox(
            "Classe do bilhete",
            [1, 2, 3],
            format_func=lambda valor: {
                1: "1 — VIP",
                2: "2 — Padrão",
                3: "3 — Econômica"
            }[valor]
        )

        traje_pressurizado = st.selectbox(
            "Vestia traje pressurizado no momento do impacto?",
            ["Sim", "Nao"]
        )

        titulo = st.selectbox(
            "Título/posto do tripulante",
            ["Cap", "Ten", "Eng", "Dra", "Passag"]
        )

    with col2:
        st.markdown("#### 👨‍🚀 Características do tripulante")

        idade = st.slider(
            "Idade",
            min_value=0,
            max_value=90,
            value=30,
            help="Selecione a idade do tripulante."
        )

        parentes_a_bordo = st.number_input(
            "Parentes a bordo",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        horas_treinamento = st.number_input(
            "Horas de treinamento de simulação espacial",
            min_value=0.0,
            max_value=300.0,
            value=50.0,
            step=5.0
        )

        tarifa_creditos = st.number_input(
            "Tarifa paga (créditos cósmicos)",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
            step=10.0
        )

    st.divider()

    enviar = st.form_submit_button(
        "🛰️ Prever resgate",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 5. PROCESSAMENTO E RESULTADO
# ==========================================

if enviar:

    # Organiza os dados no formato esperado pelo modelo.
    entrada = pd.DataFrame([{
        "modulo_estacao": modulo_estacao,
        "classe_bilhete": classe_bilhete,
        "idade": idade,
        "parentes_a_bordo": parentes_a_bordo,
        "horas_treinamento": horas_treinamento,
        "traje_pressurizado": traje_pressurizado,
        "tarifa_creditos": tarifa_creditos,
        "titulo": titulo
    }])

    try:
        # Executa a classificação.
        predicao = pipeline.predict(entrada)[0]

        # Identifica a classe correspondente a "resgatado".
        classes = list(pipeline.classes_)

        if 1 not in classes:
            st.error(
                "Não foi encontrada a classe 1 no modelo. "
                "Verifique qual valor representa o resgate "
                "no conjunto de dados utilizado no treinamento."
            )
            st.stop()

        indice_resgatado = classes.index(1)

        # Obtém a probabilidade estimada de resgate.
        probabilidade = pipeline.predict_proba(entrada)[0][
            indice_resgatado
        ]

        st.divider()
        st.header("📊 Resultado da previsão")

        resultado_col, probabilidade_col = st.columns(2)

        with resultado_col:
            if predicao == 1:
                st.success("### ✅ Resgatado")
                st.write(
                    "O modelo classificou o tripulante "
                    "na categoria de resgate."
                )
            else:
                st.error("### ❌ Não resgatado")
                st.write(
                    "O modelo classificou o tripulante "
                    "na categoria de não resgate."
                )

        with probabilidade_col:
            st.metric(
                label="Probabilidade estimada de resgate",
                value=f"{probabilidade:.1%}"
            )

            st.progress(
                float(probabilidade),
                text="Probabilidade estimada pelo modelo"
            )

        st.caption(
            "A previsão é uma estimativa produzida pelo modelo "
            "de aprendizado de máquina e não representa uma "
            "garantia de resgate real."
        )

        with st.expander("🔎 Conferir dados enviados ao modelo"):
            st.dataframe(
                entrada,
                hide_index=True,
                use_container_width=True
            )

    except Exception as erro:
        st.error(
            "Não foi possível realizar a previsão. "
            "Confira os dados e tente novamente."
        )
        st.caption(f"Detalhes técnicos: {erro}")


# ==========================================
# 6. RODAPÉ
# ==========================================

st.divider()

st.caption(
    "SpaceTec-Info EAJ | Projeto acadêmico de "
    "aprendizado de máquina e previsão de resgate."
)
