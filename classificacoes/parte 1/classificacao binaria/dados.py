import csv


def carregar_acessos():

    X = []
    Y = []

    leitor = carrega_arquivo('acesso.csv')
    for home, como_funciona, contato, comprou in leitor:
        dado = [int(home), int(como_funciona), int(contato)]
        X.append(dado)
        Y.append(int(comprou))

    return X, Y


def carrega_arquivo(filename):
    arquivo = open(filename, 'r')
    leitor = csv.reader(arquivo)
    next(leitor)
    return leitor


def carregar_buscas():

    X = []
    Y = []

    leitor = carrega_arquivo('busca.csv')
    for home, busca, logado, comprou in leitor:
        dado = [int(home), str(busca), int(logado)]
        X.append(dado)
        Y.append(int(comprou))

    return X, Y
