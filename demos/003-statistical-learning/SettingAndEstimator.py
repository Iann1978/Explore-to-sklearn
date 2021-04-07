# https://scikit-learn.org/stable/tutorial/statistical_inference/settings.html

# Statistical learning: the setting and the estimator object in scikit-learn

print('Statistical learning: the setting and the estimator object in scikit-learn')

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
print(data.shape)

print(iris.DESCR)


digits = datasets.load_digits()
print(digits.images.shape)
print(digits.images[-2])

import matplotlib.pyplot as plt
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r)
plt.show()

