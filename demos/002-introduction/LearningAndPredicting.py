# https://scikit-learn.org/stable/tutorial/basic/tutorial.html

# Learning and predicting

print('Learning and predicting')
from sklearn.datasets import load_digits
from sklearn.svm import SVC

digits = load_digits()
clf = SVC(gamma=0.001, C=100.)
clf.fit(digits.data, digits.target)

result = clf.predict(digits.data[-1:])
print(result)
print(digits.target[-1])