"""
incremental PCA learning from: https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py
hsuhui chan 2019.07.22
"""

print(__doc__)
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names
n_components = 2

pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x)

ipca = IncrementalPCA(n_components=n_components, batch_size=10)#分批处理，每次10条数据
x_ipca = ipca.fit_transform(x)

colors = ['navy','turquoise','darkorange'] #海军蓝/蓝绿色/暗橘色

for x_transformed, title ,number in [(x_pca, 'PCA', 1), (x_ipca, 'Incremental PCA', 2)]:
    plt.figure(number,figsize=(8,8))
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(x_transformed[y==i, 0], x_transformed[y==i, 1], color=color, lw=2, label=target_name)

    if 'Incremental' in title:
        err = np.abs(np.abs(x_pca)-np.abs(x_ipca)).mean()
        plt.title(title + ' of IRIS dataset\nMean absolute unsigned error %.6f' %err)
    else:
        plt.title(title + ' of IRIS dataset')
    plt.legend(loc='best', scatterpoints=1, shadow=False)
    plt.axis([-4,4, -1.5,1.5])

plt.show()
