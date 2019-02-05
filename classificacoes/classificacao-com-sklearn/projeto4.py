import pandas as pd
import numpy as np
import graphviz
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

URI = 'https://bit.ly/2MPAH1Q'
SEED = 20

np.random.seed(SEED)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)
df = pd.read_csv(URI)

renomear = {
    'mileage_per_year': 'milhas_por_ano',
    'model_year': 'ano_do_modelo',
    'price': 'preco',
    'sold': 'vendido'
}

trocar = {
    'no': 0,
    'yes': 1
}

df = df.rename(columns=renomear)
df.vendido = df.vendido.map(trocar)

ano_atual = datetime.today().year
df['idade_do_modelo'] = ano_atual - df.ano_do_modelo
df['km_por_ano'] = df.milhas_por_ano * 1.60934
df = df.drop(columns=["Unnamed: 0", "milhas_por_ano", "ano_do_modelo"], axis=1)
x = df[['preco', 'idade_do_modelo', 'km_por_ano']]
y = df['vendido']

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)
scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)


def treina_e_testa(teste_x, treino_x, treino_y, teste_y, model, nome):
    model.fit(treino_x, treino_y)
    resultado = model.predict(teste_x)

    acuracia = accuracy_score(teste_y, resultado) * 100
    print(f'A acurácia do algoritimo {nome} é de %.2f%%' % acuracia)


modelo_svc = SVC()
treina_e_testa(teste_x, treino_x, treino_y, teste_y, modelo_svc, 'SVC')

modelo_tree = DecisionTreeClassifier(max_depth=4)
treina_e_testa(raw_teste_x, raw_treino_x, treino_y, teste_y, modelo_tree, 'DecisionTreeClassifier')

dot_data = export_graphviz(modelo_tree, out_file=None, feature_names=x.columns,
                           filled=True, rounded=True, class_names=["não", "sim"])
grafico = graphviz.Source(dot_data)

grafico.view()

modelo_dummy_strat = DummyClassifier()
modelo_dummy_strat.fit(treino_x, treino_y)
acuracia_dummy = modelo_dummy_strat.score(teste_x, teste_y) * 100
print('A acurácia do Dummy Stratified foi %.2f%%' % acuracia_dummy)
