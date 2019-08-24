
#MLPClassifier分类器，损失函数是交叉熵

from sklearn.neural_network import MLPClassifier
x = [[0.,0.],[1.,1.]]
y = [0,1]
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,
                    hidden_layer_sizes=(5,2), random_state=1)
clf = clf.fit(x, y)
prediction = clf.predict([[2.,2.],[-1.,-2.]])
print(prediction,'\n')

#查看网络的权重矩阵
for coef in clf.coefs_:
    print(coef.shape)

#多标签分类问题,即一个样本可能属于多个类别，区别于多分类问题
x2 = [[0.,0.],[1.,1.]]
y2 = [[0,1],[1,1]]
clf2 = MLPClassifier(solver='lbfgs',alpha=1e-5,
                     hidden_layer_sizes=(15,), random_state=1)
clf2=clf2.fit(x2,y2)
prediction2 = clf2.predict([[1.,2.]])
print('\n',prediction2)



"""
##MLP在MNIST上的权重可视化
import  matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

#load data from https://www.openml.org/d/554
x, y = fetch_openml('mnist_784',version=1,return_X_y=True)
x = x/255.

#rescale the data, use the traditional train/test split
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,),max_iter=10, alpha=1e-4, solver='sgd',
                    verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)
mlp.fit(x_train, y_train)
print('Training set score: %f' %mlp.score(x_train, y_train))
print('Test set score: %f' % mlp.score(x_test, y_test))

fig, axes = plt.subplots(4, 4)  #fig为整个图，axes为子图集合
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):    #.T为矩阵的转置， .ravel()为数据平坦化，把数据降为1维
    ax.matshow(coef.reshape(28,28), cmap=plt.cm.gray, vmin=.5*vmin,
               vmax=.5*vmax)  #plt.matshow()用于矩阵可视化,矩阵元素对应图像像素
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

"""
