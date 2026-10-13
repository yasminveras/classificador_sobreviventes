# SpaceTec-Info EAJ
## Previsão de resgate de tripulantes

Aplicação desenvolvida para prever a possibilidade de resgate de tripulantes após um impacto de micrometeorito.

O projeto utiliza um modelo de Machine Learning treinado no Google Colab e disponibilizado em uma aplicação web feita com Streamlit.

### Tecnologias utilizadas
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

### Como executar o projeto

1. Instale as dependências com o comando:

   `pip install -r requirements.txt`

2. Execute a aplicação com:

   `streamlit run app.py`

3. Abra o endereço informado no terminal, normalmente `http://localhost:8501`.

### Arquivos principais
- `app.py`: código da aplicação Streamlit.
- `modelo_final.pkl`: modelo de Machine Learning treinado.
- `requirements.txt`: bibliotecas necessárias para executar o projeto.