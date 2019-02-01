import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

filmes_df = pd.read_csv('movies.csv')
filmes_df.columns = ["filmeId", "titulo", "genero"]
filmes = filmes_df.set_index("filmeId")

notas_df = pd.read_csv("ratings.csv")
notas_df.columns = ["usuarioId", "filmeId", "nota", "momento"]

filmes['total_de_votos'] = notas_df['filmeId'].value_counts()
filmes['media_notas'] = notas_df.groupby("filmeId").mean()["nota"]

# Collaborative Filtering x Content Filtering

# filmes_mais_votados = filmes.query("total_de_votos >= 50")
# print("Filmes em alta (collaborative filter)")
# print(filmes_mais_votados.sort_values("media_notas", ascending=False).head(10))
#
# print("filmes novos")
# print(filmes.sort_values("total_de_votos", ascending=True).head(10))
#
# meus_filmes = [1, 21, 19, 10, 11, 7, 2, 5]
# genero_ultimo_filme, nome_ultimo_filme = filmes[["genero", "titulo"]].loc[meus_filmes[-1]]
# recomendacao_ultimo_filme = filmes_mais_votados.query(f"genero=='{genero_ultimo_filme}'")
# print(f"Pq você assitiu {nome_ultimo_filme}, você pode gostar de... (content filter)")
# print(recomendacao_ultimo_filme.drop(meus_filmes, errors='ignore').sort_values("media_notas", ascending=False).head(10))

# Procurar usuários similares


def notas_do_usuario(usuario):
    notas_do_usuario = notas_df.query(f"usuarioId == {int(usuario)}")
    return notas_do_usuario[["filmeId", "nota"]].set_index("filmeId")


def distancia_de_vetores(a, b):
    return np.linalg.norm(a - b)


def distancia_de_usuarios(usuario_id1, usuario_id2, minimo=5):
    notas1 = notas_do_usuario(usuario_id1)
    notas2 = notas_do_usuario(usuario_id2)

    diferencas = notas1.join(notas2, lsuffix="_esquerda", rsuffix="_direita").dropna()

    if len(diferencas) < minimo:
        return None

    distancia = distancia_de_vetores(diferencas['nota_esquerda'], diferencas['nota_direita'])

    return [usuario_id1, usuario_id2, distancia]


def distancia_de_todos(voce_id, n=None):
    usuarios = notas_df["usuarioId"].unique()
    if n:
        usuarios = usuarios[:n]
    distancias = [distancia_de_usuarios(voce_id, usuario) for usuario in usuarios if usuario != voce_id]
    distancias = list(filter(None, distancias))
    return pd.DataFrame(distancias, columns=["voce", "outra_pessoa", "distancia"]).set_index("outra_pessoa")


def knn(voce_id, k=10, n=None):
    distancias = distancia_de_todos(voce_id, n=n).sort_values('distancia', ascending=True)
    return distancias.head(k)


def sugere_para(voce_id, lista_filmes, k=10, n=None):
    notas_de_voce = notas_do_usuario(voce_id)

    similares = knn(voce_id, k=k, n=n)
    usuarios_similares = similares.index
    notas_dos_similares = notas_df.set_index("usuarioId").loc[usuarios_similares]
    notas_dos_similares = notas_dos_similares.set_index("filmeId").drop(notas_de_voce.index, errors='ignore')
    recomendacoes = notas_dos_similares.groupby("filmeId").mean()[["nota"]]
    aparicoes = notas_dos_similares.groupby("filmeId").count()[["nota"]]

    filtro_minimo = int(k / 2)
    recomendacoes = recomendacoes.join(aparicoes, lsuffix="_media_dos_usuarios", rsuffix="_aparicoes_nos_usuarios")
    recomendacoes = recomendacoes.query(f"nota_aparicoes_nos_usuarios >= {filtro_minimo}")
    recomendacoes = recomendacoes.sort_values("nota_media_dos_usuarios", ascending=False)
    return recomendacoes.join(lista_filmes)


def novo_usuario(dados):
    notas_usuario_novo = pd.DataFrame(dados, columns=["filmeId", "nota"])
    notas_usuario_novo['usuarioId'] = notas_df['usuarioId'].max() + 1
    return pd.concat([notas_df, notas_usuario_novo], sort=True)


notas_df = novo_usuario([[122904, 2], [1246, 5], [2529, 2], [2329, 5], [2324, 5], [1, 2], [7, 0.5], [2, 2], [1196, 1], [260, 1]])

filmes_mais_votados = filmes.query("total_de_votos >= 100")

notas_df = notas_df.set_index("filmeId").loc[filmes_mais_votados.index]
notas_df = notas_df.reset_index()

recomendacoes_para_vc = sugere_para(611, filmes_mais_votados)
print(recomendacoes_para_vc.reset_index().head(10))

recomendacoes_para_vc = sugere_para(611, filmes_mais_votados, k=20)
print(recomendacoes_para_vc.reset_index()   .head(10))
