import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="SpaceTec-Info EAJ - Previsão de Resgate",
    page_icon="🛰️",
    layout="wide"
)

# Carrega o modelo treinado (modelo_final.pkl dentro da pasta models/)
CAMINHO_MODELO = Path(__file__).parent / "models" / "modelo_final.pkl"

with open(CAMINHO_MODELO, "rb") as arquivo:
    pipeline = pickle.load(arquivo)

# Título do projeto
st.title("🛰️ SpaceTec-Info EAJ")
st.subheader("Previsão de resgate de tripulantes")

st.write(
    "Preencha os dados do tripulante para estimar a probabilidade "
    "de resgate após o impacto de micrometeorito na estação."
)

# Campos do formulário
col1, col2 = st.columns(2)

with col1:
    modulo_estacao = st.selectbox(
        "Módulo da estação",
        ["Comando", "Laboratorio", "Habitacao", "Engenharia"]
    )

    classe_bilhete = st.selectbox(
        "Classe do bilhete (1 = VIP, 2 = Padrão, 3 = Econômica)",
        [1, 2, 3]
    )

    titulo = st.selectbox(
        "Título/posto do tripulante",
        ["Cap", "Ten", "Eng", "Dra", "Passag"]
    )

    traje_pressurizado = st.selectbox(
        "Vestia traje pressurizado no momento do impacto?",
        ["Sim", "Nao"]
    )

with col2:
    idade = st.slider(
        "Idade",
        min_value=0,
        max_value=90,
        value=30
    )

    parentes_a_bordo = st.number_input(
        "Parentes a bordo",
        min_value=0,
        max_value=10,
        value=0
    )

    horas_treinamento = st.number_input(
        "Horas de treinamento de simulação espacial",
        min_value=0.0,
        max_value=300.0,
        value=50.0
    )

    tarifa_creditos = st.number_input(
        "Tarifa paga (créditos cósmicos)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0
    )

# Botão de previsão
if st.button("Prever resgate", type="primary"):

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
        # Faz a previsão
        predicao = pipeline.predict(entrada)[0]

        # Obtém as classes usadas pelo modelo
        classes = list(pipeline.classes_)

        # Localiza a probabilidade correspondente à classe 1
        indice_resgatado = classes.index(1)
        probabilidade = pipeline.predict_proba(entrada)[0][indice_resgatado]

        # Exibe o resultado
        if predicao == 1:
            st.success(
                f"✅ Resgatado! Probabilidade estimada de resgate: "
                f"{probabilidade:.1%}"
            )
        else:
            st.error(
                f"❌ Não resgatado. Probabilidade estimada de resgate: "
                f"{probabilidade:.1%}"
            )

    except Exception as erro:
        st.error("Não foi possível realizar a previsão.")
        st.write("Detalhes do erro:", erro)