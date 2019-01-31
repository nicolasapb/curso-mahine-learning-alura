#!-*- coding: utf8 -*-

import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import nltk


def vetorizar_texto(texto, tradutor, stemmer):
    vetor = [0] * len(tradutor)
    for palavra in texto:
        if len(palavra) > 0:
            raiz = stemmer.stem(palavra)
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1

    return vetor


def cria_tradutor(textos, stopwords, stemmer):
    dicionario = set()
    for lista in textos:
        validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
        dicionario.update(validas)
    print(dicionario)
    print(len(dicionario))

    tuplas = list(zip(dicionario, range(len(dicionario))))
    tradutor = {palavra: indice for palavra, indice in tuplas}

    return tradutor


def fit_and_predict(modelo, nome, dados):
    k = 10
    score = cross_val_score(modelo, dados['treino_dados'], dados['treino_marcacoes'], cv=k)

    taxa_de_acerto = np.mean(score) * 100
    print(f'Taxa média de acerto do {nome}: {taxa_de_acerto}')

    return taxa_de_acerto


def analisa_arquivo(file, porcentagem_de_treino=80):

    # nltk.download('stopwords')
    # nltk.download('rslp')
    # nltk.download('punkt')

    df = pd.read_csv(file, encoding='utf-8')

    textos_puros = df['email']
    frases = textos_puros.str.lower()
    textos = [nltk.tokenize.word_tokenize(frase) for frase in frases]

    marcas = df['classificacao']

    stopwords = nltk.corpus.stopwords.words('portuguese')
    stemmer = nltk.stem.RSLPStemmer()

    tradutor = cria_tradutor(textos, stopwords, stemmer)
    vetores_de_texto = [vetorizar_texto(texto, tradutor, stemmer) for texto in textos]

    X = vetores_de_texto
    y = marcas.tolist()

    porcentagem_de_treino = porcentagem_de_treino / 100
    tamanho_de_treino = int(porcentagem_de_treino * len(y))

    treino_dados = X[0:tamanho_de_treino]
    treino_marcacoes = y[0:tamanho_de_treino]

    validacao_dados = X[tamanho_de_treino:]
    validacao_marcacoes = y[tamanho_de_treino:]

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

    dados = analisa_arquivo('emails.csv')

    # MultinomialNB
    modelo_multinomial = MultinomialNB()
    resultados[modelo_multinomial] = fit_and_predict(modelo_multinomial, "MultinomialNB", dados)

    # AdaBoostClassifier
    modelo_adaboost = AdaBoostClassifier(random_state=0)
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
