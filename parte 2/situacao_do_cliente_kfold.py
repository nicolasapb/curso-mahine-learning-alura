from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def fit_and_predict(modelo, nome, dados):
    k = 10
    score = cross_val_score(modelo, dados['treino_dados'], dados['treino_marcacoes'], cv=k)

    taxa_de_acerto = np.mean(score) * 100
    print(f'Taxa média de acerto do {nome}: {taxa_de_acerto}')

    return taxa_de_acerto


def analisa_arquivo(file, porcentagem_de_treino=80):
    df = pd.read_csv(file)
    X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
    Y_df = df['situacao']

    Xdummies_df = pd.get_dummies(X_df)
    Ydummies_df = Y_df

    X = Xdummies_df.values
    Y = Ydummies_df.values

    porcentagem_de_treino = porcentagem_de_treino / 100

    tamanho_de_treino = int(porcentagem_de_treino * len(Y))

    treino_dados = X[0:tamanho_de_treino]
    treino_marcacoes = Y[0:tamanho_de_treino]

    validacao_dados = X[tamanho_de_treino:]
    validacao_marcacoes = Y[tamanho_de_treino:]

    return {'treino_dados': treino_dados,
            'treino_marcacoes': treino_marcacoes,
            'validacao_dados': validacao_dados,
            'validacao_marcacoes': validacao_marcacoes}


def taxa_acerto_base(teste_marcacoes):
    acerto_base = max(Counter(teste_marcacoes).values())
    taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
    print(f'Taxa de acerto base: {taxa_de_acerto_base}')


def teste_real(modelo, dados):
    modelo.fit(dados['treino_dados'], dados['treino_marcacoes'])
    resultado_validacao = modelo.predict(dados['validacao_dados'])
    acertos = resultado_validacao == dados['validacao_marcacoes']
    total_de_acertos = sum(acertos)

    taxa_acerto_validacao = 100.0 * total_de_acertos / len(dados['validacao_dados'])
    print(f'taxa de acerto vencedor irl: {taxa_acerto_validacao}')
    print(f"total de elementos validados: {len(dados['validacao_dados'])}")


if __name__ == '__main__':
    resultados = {}

    dados = analisa_arquivo('situacao_do_cliente.csv')

    # MultinomialNB
    modelo_multinomial = MultinomialNB()
    resultados[modelo_multinomial] = fit_and_predict(modelo_multinomial, "MultinomialNB", dados)

    # AdaBoostClassifier
    modelo_adaboost = AdaBoostClassifier()
    resultados[modelo_adaboost] = fit_and_predict(modelo_adaboost, "AdaBoostClassifier", dados)

    # OneVsRestClassifier
    modelo_oneVsRest = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=10000))
    resultados[modelo_oneVsRest] = fit_and_predict(modelo_oneVsRest, "OneVsRestClassifier", dados)

    # OneVsOneClassifier
    modelo_oneVsOne = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
    resultados[modelo_oneVsOne] = fit_and_predict(modelo_oneVsOne, "OneVsOneClassifier", dados)

    # a eficacia do algoritimo que chuta tudo um unico valor
    taxa_acerto_base(dados['validacao_marcacoes'])

    # valida o melhor algorítimo
    teste_real(max(resultados, key=resultados.get), dados)
