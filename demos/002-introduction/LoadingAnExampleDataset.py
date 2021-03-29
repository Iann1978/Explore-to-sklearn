# https://scikit-learn.org/stable/tutorial/basic/tutorial.html

# Loading an example dataset

print('Loading an example dataset')

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
print(digits.target)
print(digits.images[0])
print(digits.data[0])