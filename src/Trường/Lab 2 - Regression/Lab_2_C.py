import matplotlib.pyplot as plt
import notebook
import numpy as np
from numpy.linalg import inv # tính nghịch đảo của ma trận

filename = 'datasets/housing-dataset.csv'
mydata = np.genfromtxt(filename, delimiter=",")

# we have n data-points (houses)
n = len(mydata)

# X is a matrix of two column, i.e. an array of n 2-dimensional data-points
X = mydata[:, :2].reshape(n, 2)

# y is the vector of outputs, i.e. an array of n scalar values
y = mydata[:, -1]

print(X.shape)
print(y.shape)

print(X[0:5])
print(y[0:5])

mu = np.mean(X, axis = 0)
print(mu)
std = np.std(X, axis = 0)
print(std)
X_normalized = (X - mu)/std

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig1.suptitle("Before normalized")
fig2.suptitle("After normalized")
ax1.scatter(X[:, 0], X[:, 1], color='red')
ax2.scatter(X_normalized[:, 0], X_normalized[:, 1], color='blue')
# plt.show()

def add_all_ones_column(X):
    n, d = X.shape
    XX = np.ones((n, d + 1))
    XX[:, 1:] = X
    return XX

X_normalized_new = add_all_ones_column(X_normalized)
print(X_normalized_new[:10])

def E(theta, X, y):
    n = len(y)
    J = np.dot(X, theta) - y
    Jt = J.T
    EE = np.dot(Jt, J) / (2 * n)
    return EE

def grad_E(theta, X, y):
    n = len(y)
    Xt = X.T
    J = np.dot(X, theta) - y
    grad = np.dot(Xt, J) / n
    return grad

def LinearRegressionWithGD(theta, alpha, max_iterations, epsilon):
    errs = []

    for itr in range(max_iterations):
        mse = E(theta, X_normalized_new, y)
        errs.append(mse)

        theta = theta - alpha*grad_E(theta, X_normalized_new, y)

        CONDITION = abs(mse - E(theta, X_normalized_new, y)) < epsilon
        if CONDITION:
            break

    return errs, theta

fig, ax = plt.subplots()
ax.set_xlabel("Number of Iterations")
ax.set_ylabel(r"Cost $E(\theta)$")

theta_init = np.array([0, 0, 0])
max_iterations = 100
epsilon = 0.000000000001

# tạo một loạt các đường đồ thị
for alpha in [0.01, 0.05, 0.1]:
    errs, theta = LinearRegressionWithGD(theta_init, alpha, max_iterations, epsilon)
    print("alpha = {}, theta = {}".format(alpha, theta))
    ax.plot(errs, label=r"With $\alpha$ = {}".format(alpha))

# add note in graph
plt.legend()
plt.show()

x = np.array([1650, 3])
xx = (x - mu)/std
xxx = add_all_ones_column(xx.reshape(1, 2)) # make the array have the same size
result = xxx@theta

print(result)

new_X = add_all_ones_column(X)
print(new_X.shape)
new_Xt = new_X.T
theta = inv(new_Xt.dot(new_X)).dot(new_Xt).dot(y)
print("With the original (non-normalized) dataset: theta = {}".format(theta))

x = np.array([1, 1650, 3])
prediction = x.dot(theta)
print(prediction)

Xt_normalized_new = X_normalized_new.T
theta = inv(Xt_normalized_new.dot(X_normalized_new)).dot(Xt_normalized_new).dot(y)
print("With the normalized dataset: theta = {}".format(theta))

x = np.array([1650, 3])
x_normalized = (x-mu)/std
x_normalized_new = add_all_ones_column(x_normalized.reshape(1, 2))
pred = x_normalized_new.dot(theta)
print("Prediction:", pred)