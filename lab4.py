import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA


pd.set_option("display.max_rows", None, "display.max_columns", None)


def visualize():
    fromage = pd.read_table(r"./fromage1.txt", sep="\t", header=0, index_col=0)
    # print(fromage.describe())
    pd.plotting.scatter_matrix(fromage, figsize=(9, 9))
    plt.show()


def kmeans(n_clusters=4):       
    fromage = pd.read_table(r"./fromage1.txt", sep="\t", header=0, index_col=0)
    np.random.seed(0)
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(fromage)
    idk = np.argsort(kmeans.labels_)
    print(pd.DataFrame(fromage.index[idk], kmeans.labels_[idk]))
    print(kmeans.transform(fromage))
    print(kmeans.transform(fromage.iloc[idk]))
    return kmeans


def kmeans_silhouette_per_n_clusters():
    # utilisation de la métrique "silhouette"
    fromage = pd.read_table(r"./fromage1.txt", sep="\t", header=0, index_col=0)
    res = np.arange(9, dtype="double")
    for k in np.arange(9):
        km = cluster.KMeans(n_clusters=k+2)
        km.fit(fromage)
        res[k] = metrics.silhouette_score(fromage, km.labels_)
    print(res)
    plt.title("silhouette")
    plt.xlabel("# of clusters")
    plt.plot(np.arange(2, 11, 1), res)
    plt.show()
    # best n_clusters for the maximal silhouette


def agglomerative_hierarchical_clustering():
    fromage = pd.read_table(r"./fromage1.txt", sep="\t", header=0, index_col=0)
    Z = linkage(fromage, method='ward', metric='euclidean')
    plt.figure(figsize=(15, 8))
    plt.title("AHC")
    plt.title('AHC avec matérialisation des 4 classes')
    dendrogram(Z, labels=fromage.index,
               orientation='left', color_threshold=255)
    print(Z)
    plt.show()
    ahc_groupes = fcluster(Z, t=255, criterion='distance')
    print(ahc_groupes)
    ordered_groups_indexes = np.argsort(ahc_groupes)
    print(pd.DataFrame(
        fromage.index[ordered_groups_indexes], ahc_groupes[ordered_groups_indexes]))



def PCA_plots():

    fromage = pd.read_table(r"./fromage1.txt", sep="\t", header=0, index_col=0)
    pca = PCA(n_components=2).fit_transform(fromage)
    

    # kmeans PCA plot
    np.random.seed(0)
    kmeans = cluster.KMeans(n_clusters=4)
    kmeans.fit(fromage)
    fig = plt.figure(figsize=(8, 8))
    kmeans_PCA_plt = fig.add_subplot(1, 1, 1)
    kmeans_PCA_plt.set_title('PCA kmeans', fontsize=20)
    for couleur, k in zip(['red', 'blue', 'lawngreen', 'aqua'], [0, 1, 2, 3]):
        kmeans_PCA_plt.scatter(pca[kmeans.labels_ == k, 0],
                               pca[kmeans.labels_ == k, 1], c=couleur)


    #AHC PCA plot
    
    Z = linkage(fromage, method='ward', metric='euclidean')
    AHC_groupes = fcluster(Z, t=255, criterion='distance')
    fig = plt.figure(figsize=(8, 8))
    AHC_PCA_plt = fig.add_subplot(1, 1, 1)
    AHC_PCA_plt.set_title('PCA AHC', fontsize=20)
    
    for couleur, k in zip(['red', 'blue', 'lawngreen', 'aqua'], [1,2,3,4]):
        AHC_PCA_plt.scatter(pca[AHC_groupes== k, 0],
                               pca[AHC_groupes == k, 1], c=couleur)


    
    
    plt.show()


# PCA_plots()
agglomerative_hierarchical_clustering()

