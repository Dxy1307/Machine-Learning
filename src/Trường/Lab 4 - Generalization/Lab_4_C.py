# Random Forest

# loading the dataset:
import numpy as np

filename = "datasets/university-admission-dataset.csv"
mydata = np.genfromtxt(filename, delimiter=',')
X = mydata[:, :2]
y = mydata[:, -1]

print(X[:3])
print(y[:2])

# using sklearn.tree.DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0).fit(X, y) # Training
preds = clf.predict(X) # Predicting

print("Predictions:", preds)

acc = np.mean(preds == y) * 100
print("Training Accuracy: {}".format(acc))

# Implementing a simplified random forest classifier
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

def train(X, y, n_features, n_samples, n_trees):
    n, d = X.shape
    clfs = []

    for itr in range(n_trees):
        ids_columns = np.random.choice(d, n_features, replace=False)
        ids_samples = np.random.choice(n, n_samples, replace=True)
        Xsub = X[ids_samples, ids_columns].reshape(-1, 1)
        ysub = y[ids_samples]
        clf = DecisionTreeClassifier(random_state=0).fit(Xsub, ysub)
        clfs.append((clf, ids_columns))

    return clfs

def predict(clfs, X):
    result = []
    for clf, index in clfs:
        preds = clf.predict(X[:, index])
        result.append(preds)

    return stats.mode(result)[0]

clfs = train(X, y, n_features=1, n_samples=50, n_trees=10)
y_pred = predict(clfs, X)
acc = np.mean(y_pred == y) * 100
print("Training Accuracy: {}".format(acc))

# Optional: Testing your implementation a dataset with multiple features