"""
#练习1，简单的决策树,二分类
from sklearn import tree
x = [[0,0],[1,1]]
y = [0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
prediction = clf.predict([[2,2]]) #预测样本类别
print(prediction)

prediction_pro = clf.predict_proba([[2,2]]) #预测每个类的概率
print(prediction_pro)


"""


#多分类决策树

from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
iris = load_iris()
x = iris.data
y = iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('iris')



