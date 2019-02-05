import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20
URI = 'https://bit.ly/2Bixmn4'

df = pd.read_csv(URI)

mapa = {
    "home": "principal",
    "how_it_works": "como_funciona",
    "contact": "contato",
    "bought": "comprou"
}
df = df.rename(columns=mapa)
df_X = df[["principal", "como_funciona", "contato"]]
df_Y = df["comprou"]

treino_X, teste_X, treino_Y, teste_Y = train_test_split(df_X, df_Y, test_size=0.24, random_state=SEED, stratify=df_Y)

print(treino_Y.value_counts())
print(teste_Y.value_counts())

modelo = LinearSVC(random_state=SEED)

modelo.fit(treino_X, treino_Y)

resultado_treino = modelo.predict(teste_X)

taxa_de_acerto = accuracy_score(teste_Y, resultado_treino) * 100

print("taxa de acerto %.2f%%" % taxa_de_acerto)
