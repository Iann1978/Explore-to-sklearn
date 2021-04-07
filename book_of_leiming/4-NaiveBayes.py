print("4-NaiveBayes")

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# 生产所有测试样本点
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, y_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# 载入鸢尾花数据    
iris = datasets.load_iris()
# 只是用前面两个特征
X = iris.data[:,:2]
# 样本标签
y = iris.target

# 创建并训练整体朴素贝叶斯分类器
clf = GaussianNB()
clf.fit(X, y)

title = ('GaussianBayesClassifier')

fig, ax = plt.subplots(figsize=(5,5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

X_pred = np.c_[xx.ravel(), yy.ravel()]
y_pred = clf.predict(X_pred)
y_pred = y_pred.reshape(xx.shape)

xxx = xx[0]
yyy = yy[:, 0]
ax.contourf(xxx, yyy, y_pred, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()


