import matplotlib.pyplot as plt
import notebook
import math
import numpy as np

filename = 'datasets/ex1data2.txt'
mydata = np.genfromtxt(filename, delimiter=",")

# we have n data-points
n = len(mydata)

# X is a matrix of two column, i.e. an array of n 2-dimensional data-points
X = mydata[:, :2].reshape(n, 2)

# y is the vector of outputs, i.e. an array of n scalar values
y = mydata[:, -1]

# print(X[:10])
# print(y[:10])

def gaussian_kernel(u, v, gamma):
    d = np.linalg.norm(u - v) # compute distance between 2 vectors
    hat = -(d**2)*gamma
    k = math.exp(hat)
    return k

def h(x, X, y, gamma):
    # An array containing the similarity between x and all the others data-points in X:
    similarities = np.array([gaussian_kernel(x, xi, gamma) for xi in X])
    k = similarities@y
    co = similarities.sum()
    if co == 0:
        print("Error")
    return k/co

gamma = 0.00005
x = np.array([1650, 3])
prediction = h(x, X, y, gamma)
print("The prediction on x is:", prediction)

x = np.array([1650, 3])
gammas_list = np.arange(1e-10, 10e-5, 1e-5)
predictions = []

for gamma in gammas_list:
    prediction = h(x, X, y, gamma)
    predictions.append(prediction)

fig, ax = plt.subplots()
ax.plot(gammas_list, predictions)
# plt.show()

gamma = 0.00005
n = len(X)
X_train = X[ : n//2]
y_train = y[ : n//2]

X_test = X[n//2 : ]
y_test = y[n//2 : ]

y_pred = np.array([h(xi, X_train, y_train, gamma) for xi in X_test])

fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], y_test, color='red', label="True Price")
ax.scatter(X_test[:, 0], y_pred, color='blue', label="Predicted Price")
ax.set_xlabel("House size")
ax.set_ylabel("House price")
plt.legend()
plt.show()