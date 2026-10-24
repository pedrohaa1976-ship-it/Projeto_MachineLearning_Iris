import streamlit as st
import joblib
modelo = joblib.load('modelos/knn_flores_iris.joblib')
# Configuração inicial da página
st.set_page_config(
    page_title="Classificador de Flores Iris",
    page_icon="🌸",
    layout="centered"
)

# Título principal
st.title("🌸 Classificador de Flores Iris")

st.write(
    """
    Este aplicativo recebe as características de uma flor Iris e,
    futuramente, usará um modelo de Machine Learning para tentar adivinhar
    a espécie da planta.
    """
)

st.divider()

# Seção explicativa
st.header("Informe as características da flor")

st.info(
    """
    Preencha os valores abaixo com as medidas da flor em centímetros.
    
    O modelo Iris normalmente utiliza quatro características:
    
    - Comprimento da sépala
    - Largura da sépala
    - Comprimento da pétala
    - Largura da pétala
    """
)

# Criando duas colunas para deixar o layout mais bonito
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Comprimento da sépala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Comprimento da pétala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Largura da sépala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Largura da pétala (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

st.divider()

# Mostrando os dados informados pelo usuário
st.subheader("Dados informados")

dados_iris = {
    "Comprimento da sépala": sepal_length,
    "Largura da sépala": sepal_width,
    "Comprimento da pétala": petal_length,
    "Largura da pétala": petal_width
}

st.write(dados_iris)
botao_adivinhar = st.button("Adivinhar espécie da flor", type="primary")

if botao_adivinhar:

    entrada_modelo = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    previsao = modelo.predict(entrada_modelo)
    if previsao[0] == 0:
        especie = "Iris Setosa"
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeOtARnk4DrvIQz7LBLVNDhO92c-yMzpE2CffNbdhYBw&s=10", caption="Iris Setosa")
    elif previsao[0] == 1:  
        especie = "Iris Versicolor"
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRPevcNjSy2TPFzTV4PZwkmKcZt2zHos8g6k-vNpHOpFQ&s=10", caption="Iris Versicolor")
    else:
        especie = "Iris Virginica"
        st.image("https://as1.ftcdn.net/v2/jpg/03/15/03/12/1000_F_315031217_fQSpGxilJ7yVCmPn3hyvrTFi0TmSIekn.jpg", caption="Iris Virginica")

    st.success(f"**Resultado da Previsão:** A espécie da flor é **{especie}**!")
