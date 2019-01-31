import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter

# Naiave Bayes
# 1. 'home', 'busca', 'logado' 90% para treino e 10% para teste: 75.00%
# 2. 'home', 'busca' 90% para treino e 10% para teste: 75.00%
# 3. 'home', 'logado' 90% para treino e 10% para teste: 62.50%
# 4. 'busca', 'logado' 90% para treino e 10% para teste: 75.00%
# 5. 'busca'  90% para treino e 10% para teste: 75.00%
# 6. 'home'  90% para treino e 10% para teste: 62.50%
# 7. 'logado'  90% para treino e 10% para teste: 62.50%
# AdaBoost

def fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):

    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    acertos = resultado == teste_marcacoes
    total_de_acertos = sum(acertos)

    return 100.0 * total_de_acertos / len(teste_dados)


def analisa_arquivo(file, porcentagem_de_treino=80, porcentagem_de_teste=10):

    df = pd.read_csv(file)
    X_df = df[['home', 'busca', 'logado']]
    Y_df = df['comprou']

    Xdummies_df = pd.get_dummies(X_df)
    Ydummies_df = Y_df

    X = Xdummies_df.values
    Y = Ydummies_df.values

    porcentagem_de_treino = porcentagem_de_treino / 100
    porcentagem_de_teste = porcentagem_de_teste / 100

    tamanho_de_treino = int(porcentagem_de_treino * len(Y))
    tamanho_de_teste = int(porcentagem_de_teste * len(Y))
    # tamanho_de_validacao = int(len(Y) - tamanho_de_treino - tamanho_de_teste)

    treino_dados = X[0:tamanho_de_treino]
    treino_marcacoes = Y[0:tamanho_de_treino]

    fim_de_teste = tamanho_de_treino + tamanho_de_teste
    teste_dados = X[tamanho_de_treino:fim_de_teste]
    teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

    validacao_dados = X[fim_de_teste:]
    validacao_marcacoes = Y[fim_de_teste:]

    return treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, validacao_dados, validacao_marcacoes


def taxa_acerto_base(teste_marcacoes):
    acerto_base = max(Counter(teste_marcacoes).values())
    taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
    print(f'Taxa de acerto base: {taxa_de_acerto_base}')


if __name__ == '__main__':

    treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, validacao_dados, validacao_marcacoes\
        = analisa_arquivo('busca2.csv')

    modelo_multinomial = MultinomialNB()

    resultado_multinomial = fit_and_predict(modelo_multinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

    print(f'Taxa de acerto do MultinomialNB: {resultado_multinomial}')

    modelo_adaboost = AdaBoostClassifier()

    resultado_adaboost = fit_and_predict(modelo_adaboost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

    print(f'Taxa de acerto do AdaBoost: {resultado_adaboost}')

    # a eficacia do algoritimo que chuta tudo um unico valor
    taxa_acerto_base(validacao_marcacoes)

    print(f'total de elementos testados: {len(teste_dados)}')

    if resultado_adaboost > resultado_multinomial:
        vencedor = modelo_adaboost
    else:
        vencedor = modelo_multinomial

    resultado_validacao = vencedor.predict(validacao_dados)
    acertos = resultado_validacao == validacao_marcacoes
    total_de_acertos = sum(acertos)

    taxa_acerto_validacao = 100.0 * total_de_acertos / len(validacao_dados)
    print(f'taxa de acerto vencedor irl: {taxa_acerto_validacao}')


