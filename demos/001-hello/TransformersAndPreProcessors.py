# https://scikit-learn.org/stable/getting_started.html#pipelines-chaining-pre-processors-and-estimators

# Transformers and pre-processors

print('Transformers and pre-processors')

from sklearn.preprocessing import StandardScaler
X = [[0, 15],
     [1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)
