import pandas as pd
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

pd.set_option('display.max_columns', 520)
pd.set_option('display.width', 1000)
URI_FILMES = 'https://raw.githubusercontent.com/oyurimatheus/clusterirng/master/movies/movies.csv'

filmes = pd.read_csv(URI_FILMES)
filmes.columns = ["filme_id", "titulo", "generos"]
generos = filmes.generos.str.get_dummies()
# dados = filmes.join(dummies_generos)
# print(dados.head(10))

dados_dos_filmes = pd.concat([filmes, generos], axis=1)

scaler = StandardScaler()
generos_escalados = scaler.fit_transform(generos)


# def kmeans(numero_de_clusters, generos):
#     modelo = KMeans(n_clusters=numero_de_clusters, random_state=0)
#     modelo.fit(generos)
#     return [numero_de_clusters, modelo.inertia_]


# a = [kmeans(numero_de_grupos, generos_escalados) for numero_de_grupos in range(1, 100)]
# a = pd.DataFrame(a, columns=['grupos', 'inertia'])
#
# a.inertia.plot(xticks=a.grupos)

modelo = KMeans(n_clusters=16, random_state=0)
modelo.fit(generos_escalados)

grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
# grupo = 10
# filtro = modelo.labels_ == grupo
# print(dados_dos_filmes[filtro].head())

# modelo = AgglomerativeClustering(n_clusters=16)
# grupos = modelo.fit_predict(generos_escalados)

# grupo = 0
# filtro = modelo.labels_ == grupo
# print(dados_dos_filmes[filtro].sample(10))

matriz_de_distancia = linkage(grupos)
dendrograma = dendrogram(matriz_de_distancia) 
print(dendrograma)