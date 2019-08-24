# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:37:21 2019

@author: HsuhuiChan
PCA and LDA learning
"""
print(__doc__)
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components = 2)
x_r = pca.fit(x).transform(x)

lda = LinearDiscriminantAnalysis(n_components=2)
x_r2 = lda.fit(x, y).transform(x)

# Percentage of variance explained for each components
print('explained variance ratio of PCA (first two components): %s' % str(pca.explained_variance_ratio_))
print('explained variance ratio of LDA (first two components): %s' % str(lda.explained_variance_ratio_))
plt.figure(1)
colors = ['navy','turquoise','darkorange']
lw = 2

for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(x_r[y == i, 0], x_r[y == i, 1], color=color, alpha = 0.8, lw = lw, label = target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')


plt.figure(2)
for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(x_r2[y == i, 0], x_r2[y == i, 1], color=color, alpha=0.8, lw=lw, label = target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()