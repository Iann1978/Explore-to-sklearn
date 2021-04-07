import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree


# 生成所有测试样本点
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# 载入iris数据集
iris = datasets.load_iris()
# 只使用前面连个特征
X = iris.data[:, :2]
# 样本标签值
y = iris.target

# 创建并训练决策树
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)

# 创建测试集
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)

# Show Result
title = ('DecisionTreeClassifier')
fig, ax = plt.subplots(figsize = (5, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
ax.contourf(xx, yy, zz,cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()