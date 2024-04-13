# Classification with logistic regression
import matplotlib.pyplot as plt
import notebook
import numpy as np
import scipy.optimize as op

# loading the data from the file into mydata
filename = "datasets/university-admission-dataset.csv"
mydata = np.genfromtxt(filename, delimiter=",")

# We have n students
n = len(mydata)

X = mydata[:, :2]
y = mydata[:, -1]

# Visualizing the data
X0 = X[y == 0]
X1 = X[y == 1]

fig, ax = plt.subplots()
ax.scatter(X0[:,0], X0[:, 1], color="red", marker="$0$", label="Not admitted")
ax.scatter(X1[:,0], X1[:, 1], color="blue", marker="$1$", label="Admitted")
ax.set_xlabel("Exam 1 score")
ax.set_ylabel("Exam 2 score")
fig.suptitle("Scatter plot of training data")
plt.legend()
# plt.show()

# adding a first column of ones to the dataset
def add_all_ones_column(X):
    n, d = X.shape
    XX = np.ones((n, d + 1))
    XX[:, 1:] = X
    return XX

X_new = add_all_ones_column(X)

# sigmoid function
def sigmoid(z):
    e = np.exp(-z)
    denominator = 1 + e
    return 1 / denominator

# print(sigmoid(np.array([[-43, 0, 32], [4, 5, 6]])))
# print(sigmoid(np.array([4, 10, 99])))
# print(sigmoid(0))
# print(sigmoid(999999))

# the hypothesis function is defined as follows:
def h(theta, x):
    return sigmoid(theta.T @ x)

def h_all(theta, X):
    g = X@theta
    return sigmoid(g)

# Cost function and gradient
def E(theta, X, y):
    n = len(y)
    E = -y@np.log(h_all(theta, X)) - (1-y)@np.log(1 - h_all(theta, X))
    EE = np.sum(E)
    return EE / n

theta = np.array([0, 0, 0])
# print(E(theta, X_new, y))

def gradE(theta, X, y):
    n = len(y)
    grad = (h_all(theta, X) - y) @ X
    return grad / n

# learning parameters using scipy.optimize.minimize
theta = np.array([0, 0, 0])
print("Initial cost:", E(theta, X_new, y))

res = op.minimize(E, theta, (X_new, y), "TNC", gradE)
theta = res.x
print(E(theta, X_new, y))

# Plotting the decision boundary
def plot_decision_boundary(X, y, theta):
    X0 = X[y == 0]
    X1 = X[y == 1]

    fig, ax = plt.subplots()

    # Plotting the dataset:
    ax.scatter(X0[:, 0], X0[:, 1], marker="o", color="red", label="Non admitted")
    ax.scatter(X1[:, 0], X1[:, 1], marker="*", color="blue", label="Admitted")
    ax.set_xlabel("Exam 1 score")
    ax.set_ylabel("Exam 2 score")

    # Plotting the decision boundary:
    plot_x1 = np.arange(30, 100)
    plot_x2 = -(theta[0] + theta[1] * plot_x1) / theta[2]
    ax.plot(plot_x1, plot_x2, color="green", label="Decision boundary")

    ax.set_title("Plot of the training data and decision boundary")
    plt.legend()
    plt.show()

# plot_decision_boundary(X, y, theta)

# Evaluating the logistic regression model
x = np.array([1, 45, 85])

print("Admission probability of this student x ...")
print(h(theta, x))

from sklearn.metrics import accuracy_score
def predict(theta, X):
    n, d = X.shape
    result = []
    for i in range(n):
        pre = h(theta, X[i])
        output = 0
        if pre >= 0.5:
            output = 1
        else: output = 0
        result.append(output)
    return result

y_pred = predict(theta, X_new)
accuracy = accuracy_score(y, y_pred)
print("Accuracy\n", 100 * accuracy)