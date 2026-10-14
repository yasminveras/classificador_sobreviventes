import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

# Carrega o dataset (mesmo arquivo usado no Colab, agora dentro do repositório)
df = pd.read_csv('datasets/estacao_spacetecinfo.csv')

# Trata nulos (igual à seção 2.1 do notebook)
df['idade'] = df['idade'].fillna(df['idade'].median())

# Cria a feature 'titulo' a partir do nome (igual à seção 2.2 do notebook)
df['titulo'] = df['nome_tripulante'].str.extract(r'^([^.]+)\.')

# Remove colunas irrelevantes (igual à seção 1.7 do notebook)
df = df.drop(columns=['id_passageiro', 'nome_tripulante'], errors='ignore')

X = df.drop(columns=['resgatado'])
y = df['resgatado']

# Colunas que o pipeline vai receber "cruas" (sem escalonamento manual prévio)
categorical_features = ['modulo_estacao', 'traje_pressurizado', 'titulo']
numeric_features = ['idade', 'parentes_a_bordo', 'horas_treinamento', 'tarifa_creditos']
# 'classe_bilhete' é ordinal e passa direto (remainder='passthrough')

preprocessador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

# Pipeline único: pré-processamento + melhor modelo, com os hiperparâmetros
# já encontrados no Colab via GridSearchCV (seção 3.4 do notebook)
pipeline_final = Pipeline(steps=[
    ('preprocessador', preprocessador),
    ('classificador', SVC(C=10, gamma='auto', kernel='rbf', probability=True, random_state=42))
])

# Treina o pipeline inteiro com todos os dados disponíveis
pipeline_final.fit(X, y)

# Salva o pipeline treinado direto na pasta models/, igual o professor faz
with open('models/modelo_final.pkl', 'wb') as f:
    pickle.dump(pipeline_final, f)

print("Modelo treinado e salvo em models/modelo_final.pkl")