import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 20
URI = 'https://bit.ly/2MRQnSg'

np.random.seed(SEED)
df = pd.read_csv(URI)

mapa = {
    'expected_hours': 'horas_esperadas',
    'price': 'preco',
    'unfinished': 'nao_finalizado'
}
df = df.rename(columns=mapa)

troca = {1: 0, 0: 1}

df['finalizado'] = df.nao_finalizado.map(troca)

# sns.relplot(x="horas_esperadas", y="preco", data=df, col="finalizado", hue="finalizado")


x = df[['horas_esperadas', 'preco']]
y = df['finalizado']

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

model = SVC()
model.fit(treino_x, treino_y)
resultado = model.predict(teste_x)

acuracia = accuracy_score(teste_y, resultado) * 100
baseline = np.ones(len(teste_y))
acuracia_base = accuracy_score(teste_y, baseline) * 100
print('A acurácia foi de %.2f%%' % acuracia)
print('A acurácia da baseline foi de %.2f%%' % acuracia_base)

# sns.scatterplot(x="horas_esperadas", y="preco", data=teste_x, hue=teste_y)
x_min = teste_x[:, 0].min()
x_max = teste_x[:, 0].max()

y_min = teste_x[:, 1].min()
y_max = teste_x[:, 1].max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)

pontos = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(pontos)

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(teste_x[:, 0], teste_x[:, 1], c=teste_y, s=1)
plt.show()
