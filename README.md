# SpaceTec-Info EAJ — Previsão de Resgate de Tripulantes

Projeto da disciplina de Programação Avançada de IA. Modelo de classificação
treinado para prever se um tripulante da estação SpaceTec-Info EAJ foi
resgatado (1) ou perdido (0) após o impacto de micrometeorito.

## Resultados do treinamento

- Modelo selecionado: **SVM (Support Vector Machine)**
- Hiperparâmetros ótimos (GridSearchCV): `C=10`, `gamma=auto`, `kernel=rbf`
- Métricas no conjunto de teste (20%):
  - Acurácia: 0.88
  - Precisão: 0.8958
  - Recall: 0.86
  - F1-score: 0.8776

## Estrutura do projeto

```
.
├── datasets/
│   └── estacao_spacetecinfo.csv   # dataset fornecido pelo professor
├── models/
│   └── modelo_final.pkl           # gerado por treinar_modelo.py (não vem pronto)
├── app.py                         # Aplicação Streamlit
├── treinar_modelo.py              # Treina o pipeline e salva o .pkl em models/
├── requirements.txt               # Dependências do projeto
└── README.md
```

## Como rodar localmente

1. Criar e ativar o ambiente virtual:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Instalar as dependências:
   ```
   pip install -r requirements.txt
   ```
3. Treinar o modelo (gera `models/modelo_final.pkl`):
   ```
   python treinar_modelo.py
   ```
4. Rodar o app:
   ```
   streamlit run app.py
   ```

## Como subir para o GitHub

```
git init
git add .
git commit -m "Projeto SpaceTec-Info EAJ - app Streamlit"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
git push -u origin main
```