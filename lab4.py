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
    dendrogram(Z, labels=range(29),
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

    # AHC PCA plot

    Z = linkage(fromage, method='ward', metric='euclidean')
    AHC_groupes = fcluster(Z, t=255, criterion='distance')
    fig = plt.figure(figsize=(8, 8))
    AHC_PCA_plt = fig.add_subplot(1, 1, 1)
    AHC_PCA_plt.set_title('PCA AHC', fontsize=20)

    for couleur, k in zip(['red', 'blue', 'lawngreen', 'aqua'], [1, 2, 3, 4]):
        AHC_PCA_plt.scatter(pca[AHC_groupes == k, 0],
                            pca[AHC_groupes == k, 1], c=couleur)

    plt.show()


def DIANA_based_on_KMeans():

    def split(group):
        np.random.seed(0)
        kmeans = cluster.KMeans(n_clusters=2).fit(group)
        cluster0 = group[kmeans.labels_ == 0]
        cluster1 = group[kmeans.labels_ == 1]
        # print(kmeans.cluster_centers_[0])
        distance = metrics.pairwise.euclidean_distances(
            [kmeans.cluster_centers_[0]], [kmeans.cluster_centers_[1]])
        return [cluster0, cluster1, distance[0][0]]

    def adjust_linkage_pointers(linkage, n_samples):
        adjusted_linkage = []
        for i, [index0, index1, distance, size] in enumerate(linkage):
            if(index0 < 0):
                index0 = n_samples+i+index0
            if(index1 < 0):
                index1 = n_samples+i+index1
            adjusted_linkage += [[float(index0), float(index1),
                                  float(distance), float(size)]]

        return adjusted_linkage

    fromage = pd.read_table(r"./fromage1.txt", sep="\t", header=0, index_col=0)
    groups = [fromage]

    linkage = []
    maxDistance = 60
    i = 0
    while(i < len(groups)):
        currentGroup = groups[i]
        [cluster0, cluster1,distance] = split(currentGroup)

        if(len(cluster0) == 1):
            index0 = fromage.index.get_loc(cluster0.index[0])
            if(len(cluster1) == 1):
                index1 = fromage.index.get_loc(cluster1.index[0])
            else:
                index1 = i-len(groups)
                groups += [cluster1]
        else:
            index0 = i-len(groups)
            if(len(cluster1) == 1):
                index1 = fromage.index.get_loc(cluster1.index[0])
            else:
                index1 = i-len(groups)-1
                groups += [cluster1]
            groups += [cluster0]

        # distance = maxDistance-2*i
        size = len(currentGroup)
        linkage.insert(0, [index0, index1, distance, size])

        i += 1

    linkage = adjust_linkage_pointers(linkage, len(fromage))
    dendrogram(linkage, labels=fromage.index,
               orientation='left', color_threshold=255)

    plt.show()


DIANA_based_on_KMeans()


# print( np.argsort(km1.labels_))
# indexes0=fromage.index[km1.labels_==0]
# indexes0=fromage.loc[['CarredelEst']]
# indexes0=fromage.index.get_loc('CarredelEst')
# print(indexes0)
# print(metrics.silhouette_samples(fromage, km1.labels_))
# km2 = cluster.KMeans(n_clusters=2).fit(fromage[km1.labels_ == 0])
# print( np.argsort(km2.labels_))

# PCA_plots()
# agglomerative_hierarchical_clustering()
# Z=[
#     [0.0, 3.0, 1.0, 2.0],
#     [1.0, 5.0, 2.0, 3.0],
#     [2.0, 4.0, 3.0, 2.0],
#     [6.0, 7.0, 4.0, 4.0]
# ]

# inverted=[
#     []
# ]

# dendrogram(inverted[::0])
# plt.show()
