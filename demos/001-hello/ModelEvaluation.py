# https://scikit-learn.org/stable/getting_started.html#pipelines-chaining-pre-processors-and-estimators
# Mode evaluation

print('Mode evaluation')

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)
print(result['test_score'])
