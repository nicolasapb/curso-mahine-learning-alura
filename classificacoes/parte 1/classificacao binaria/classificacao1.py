from sklearn.naive_bayes import MultinomialNB

p1 = [1, 1, 0]
p2 = [1, 1, 0]
p3 = [1, 1, 0]
c1 = [1, 1, 1]
c2 = [0, 1, 1]
c3 = [0, 1, 1]

dados = [p1, p2, p3, c1, c2, c3]
marcacoes = [1, 1, 1, -1, -1, -1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]
testes = [misterioso1, misterioso2, misterioso3]
marcacoes_teste = [-1, 1, -1]

resultado = modelo.predict(testes)
print(resultado)

diferencas = resultado - marcacoes_teste
print(diferencas)

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(testes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
print(taxa_de_acerto)

