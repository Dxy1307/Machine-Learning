# Classification with KNN
from tkinter import W
import matplotlib.pyplot as plt
import notebook
import numpy as np

# loading the data
filename = "datasets/microchips-dataset.csv"
mydata = np.genfromtxt(filename, delimiter=",")

n = len(mydata)
X = mydata[:, :2]
y = mydata[:, -1]

# Visualizing the data
X0 = X[y == 0]
X1 = X[y == 1]

fig, ax = plt.subplots()
ax.scatter(X0[:, 0], X0[:, 1], color="red", marker="x", label="Rejected (y=0)")
ax.scatter(X1[:, 0], X1[:, 1], color="blue", marker="+", label="Accepted (y=1)")
ax.set_xlabel("Microchip Test 1")
ax.set_ylabel("Microchip Test 2")
fig.suptitle("Plot of the training dataset")

plt.legend()
# plt.show()

# Implementing the k Nearest Neighbours (KNN)
from collections import Counter
def dist(u, v):
    return np.linalg.norm(u - v)

def prediction(x, X, y, k = 5):
    # Compute the list of distances from x to each point in X
    distances = [dist(x, X[i]) for i in range(n)]
    # Get the list of indices sorted according to their corresponding distances
    distances = np.array(distances)
    indices = np.argsort(distances)
    # Take the class-labels corresponding to the first k indices (closest points to x)
    k_class_labels = y[indices][:k]
    # The predicted class-label is the most common class-label one among these class-labels
    return Counter(k_class_labels).most_common(k)[0][0]

x = np.array([0, 0])
print(prediction(x, X, y, k = 5))

# Plotting the decision boundary
from matplotlib.colors import ListedColormap
def plot_decision_boundary(func, X, y, k):
    print("Please wait. This might take few seconds to plot ...")
    min_x1, max_x1 = min(X[:, 0]) - 0.1, max(X[:, 0]) + 0.1
    min_x2, max_x2 = min(X[:, 1]) - 0.1, max(X[:, 1]) + 0.1

    plot_x1, plot_x2 = np.meshgrid(np.linspace(min_x1, max_x1, 50), np.linspace(min_x2, max_x2, 50))
    points = np.c_[plot_x1.ravel(), plot_x2.ravel()]
    preds = np.array([func(x, X, y, k) for x in points])
    preds = preds.reshape(plot_x1.shape)

    X0 = X[y == 0]
    X1 = X[y == 1]

    fig, ax = plt.subplots()
    ax.pcolormesh(plot_x1, plot_x2, preds, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X0[:, 0], X0[:, 1], color="red", label="Rejected")
    ax.scatter(X1[:, 0], X1[:, 1], color="blue", label="Accepted")
    ax.set_xlabel("Microchip Test 1")
    ax.set_ylabel("Microchip Test 2")
    ax.set_title("Decision boundary for k = {}".format(k))
    plt.legend()
    plt.show()

# plot_decision_boundary(prediction, X, y, k = 1)
# plot_decision_boundary(prediction, X, y, k = 15)
# plot_decision_boundary(prediction, X, y, k = 30)
    
# Evaluating the kNN classifier
from sklearn.metrics import accuracy_score
y_pred = np.array([prediction(x, X, y, k = 15) for x in X])
accuracy = accuracy_score(y, y_pred)
print("Accuracy of the kNN classifier: ", accuracy * 100)

# Optional: Weighted kNN classifier
def prediction_weighted(x, X, y, k=5):
    # Compute the list of distances from x to each point in X
    distances, weights = [], []
    for i in range(n):
        distances.append(dist(x, X[i]))
        weights.append(1 / dist(x, X[i]))

    # Get the list of indices sorted according to their corresponding distances
    distances = np.array(distances)
    weights = np.array(weights)
    indices = np.argsort(distances)

    # Take the class-labels corresponding to the first k indices (closest points to x)
    k_class_labels = y[indices][:k]
    k_weights = weights[indices][:k]

    # The predicted class-label is the most common class-label one among these class-labels
    w0, w1 = 0, 0 # sum of weights for output 0 and 1
    for i in range(k):
        if k_class_labels[i] == 0:
            w0 += k_weights[i]
        else: w1 += k_weights[i]
    
    return 0 if w0 > w1 else 1

x = np.array([0, 0])
print(prediction_weighted(x, X, y, k = 5))

plot_decision_boundary(prediction_weighted, X, y, k = 1)
plot_decision_boundary(prediction_weighted, X, y, k = 15)
plot_decision_boundary(prediction_weighted, X, y, k = 30)