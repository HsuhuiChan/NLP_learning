"""
#决策树回归
#简单的例子

from sklearn import tree
x = [[0,0],[2,2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(x, y)
prediction = clf.predict([[1, 1]])
print(prediction)

"""

# import the necessary module and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#create a random dataset
rng = np.random.RandomState(1) #RandomState为伪随机数生成器
x = np.sort(5*rng.rand(80,1),axis=0) #生成5*[0,1]的伪随机数， axis=0表示对行在列的方向上进行排列
y = np.sin(x).ravel() #ravel()函数类似于flatten()函数，将数据降维，按行序优先并保留所有的数据
y[::5] += 3*(0.5 - rng.rand(16)) #[::5]表示每隔5个元素，进行一次操作

#fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1 = regr_1.fit(x, y)
regr_2 = regr_2.fit(x, y)

#predict
x_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis] #np.newaxis表示在该位置增加一维，由原来的[500,0],变成[500,1]
y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)

#plot the results
plt.figure()
plt.scatter(x,y,s=20,edgecolor='black',c='darkorange',label='data')
plt.plot(x_test,y_1,color='cornflowerblue',label='max_depth=2',linewidth=2)
plt.plot(x_test,y_2,color='yellowgreen',label='max_depth=5',linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()

