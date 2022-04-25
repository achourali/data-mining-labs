import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


pd.set_option("display.max_rows", None, "display.max_columns", None)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
df = pd.read_csv(url, names=features+['target'])


# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, ['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=[
                           'principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis=1)

# visualising finalDf
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
ax.legend(targets)
ax.grid()
print(pca.explained_variance_ratio_)
plt.show()



# # Standardizing the features
# x = StandardScaler().fit_transform(x)
# pca = PCA(n_components=1)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data=principalComponents,
#                            columns=['principal component 1'])
# finalDf = pd.concat([principalDf, df[['target']]], axis=1)
# print(finalDf)

# # visualising finalDf
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets, colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
#                finalDf.loc[indicesToKeep, 'principal component 1'], c=color, s=50)
# ax.legend(targets)
# ax.grid()
# print(pca.explained_variance_ratio_)
# plt.show()
