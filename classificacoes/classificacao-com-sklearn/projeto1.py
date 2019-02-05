import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features (1 sim, 0 não)
# pelo longo?
# perna curta?
# faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 => porco, 0 => cachorro
treino_X = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_Y = [1, 1, 1, 0, 0, 0]

model = LinearSVC(random_state=0)
model.fit(treino_X, treino_Y)

misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

testes_X = [misterio1, misterio2, misterio3]
testes_resultado = model.predict(testes_X)

testes_Y = [0, 1, 1]
taxa_de_acerto = accuracy_score(testes_Y, testes_resultado)

print(f'Taxa de acerto: {taxa_de_acerto * 100}')
