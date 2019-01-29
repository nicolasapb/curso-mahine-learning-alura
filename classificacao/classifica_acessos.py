from classificacao.dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

# minha abordagem inicial foi
# 1. separar 90% para treino e 10% para teste: 88.89%
# 2. separar 34% para treino e 66% para teste: 95.39%

if __name__ == '__main__':

    X, Y = carregar_acessos()

    treino_dados = X[:34]
    treino_marcacoes = Y[:34]

    teste_dados = X[-65:]
    teste_marcacoes = Y[-65:]

    modelo = MultinomialNB()
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    diferencas = resultado - teste_marcacoes
    print(diferencas)

    acertos = [d for d in diferencas if d == 0]
    total_de_acertos = len(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    print(taxa_de_acerto)
    print(total_de_elementos)
