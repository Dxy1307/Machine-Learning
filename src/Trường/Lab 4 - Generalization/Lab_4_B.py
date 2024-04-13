# Regularized Logistic Regression

# loading the data
import numpy as np
import matplotlib.pyplot as plt

filename = "datasets/microchips-dataset.csv"
mydata = np.genfromtxt(filename, delimiter=',')
n = len(mydata)
X = mydata[:, :2]
y = mydata[:, -1]

X1 = X[y == 1]
X0 = X[y == 0]
fig, ax = plt.subplots()
fig.suptitle('Plot of the training dataset')
ax.scatter(X0[:, 0], X0[:, 1], color='red', marker='x', label='Rejected')
ax.scatter(X1[:, 0], X1[:, 1], color='blue', marker='+', label='Accepted')
ax.set_xlabel('Microchip Test 1')
ax.set_ylabel('Microchip Test 2')
ax.legend()
# plt.show()

# feature mapping
def mapFeature(X1, X2):
    degree = 6
    col = 0
    out = np.ones((len(X1), 27))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out[:, col] = (X1 ** (i - j)) * (X2 ** j)
            col += 1

    out = np.c_[np.ones(len(out)), out]
    return out

X_poly = mapFeature(X[:, 0], X[:, 1])
print("Shape of X_poly is:", X_poly.shape)
print(X_poly[:3])

# Cost function and gradient
def sigmoid(z):
    e = np.exp(-z)
    return 1 / (1 + e)

def E(theta, X, y, lmd):
    n, d = X.shape
    E0 = -y@np.log(sigmoid(X@theta)) - (1-y)@np.log(1 - sigmoid(X@theta))
    E1 = E0.sum() / n
    regularized = np.square(theta[1:]).sum() * lmd / (2*n)
    return E1 + regularized

def gradE(theta, X, y, lmd):
    n, d = X.shape
    grad0 = (sigmoid(X@theta) - y)@X / n
    gradj = grad0 + lmd*theta/n
    grad0[1:] = gradj[1:]
    return grad0

theta_init = np.zeros(X_poly.shape[1])
print("Cost:", E(theta_init, X_poly, y, 0))
print("Gradient vector:", gradE(theta_init, X_poly, y, 0))

# learning parameters using scipy.optimize.minimize(...)
import scipy.optimize as op

def trainLogisticReg(X, y, theta_init, lmd):
    res = op.minimize(E, theta_init, (X, y, lmd), 'TNC', gradE)
    return res.x

theta_init = np.zeros(X_poly.shape[1])
lmd = 0

print("Initial cost:", E(theta_init, X_poly, y, lmd))
theta = trainLogisticReg(X_poly, y, theta_init, lmd)
print("Final cost:", E(theta, X_poly, y, lmd))

# plotting the decision boundary
def trainAndPlotDecisionBoundary(X_original, X_poly, y, theta_init, lmd):
    theta = trainLogisticReg(X_poly, y, theta_init, lmd)

    fig, ax = plt.subplots()
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')

    X0 = X_original[y == 0]
    X1 = X_original[y == 1]
    ax.scatter(X0[:, 0], X0[:, 1], color='red', marker='x', label='Rejected (y=0)')
    ax.scatter(X1[:, 0], X1[:, 1], color='blue', marker='+', label='Accepted (y=1)')

    x1_plot = np.linspace(-1, 1.5, 50)
    x2_plot = np.linspace(-1, 1.5, 50)
    X1_plot, X2_plot = np.meshgrid(x1_plot, x2_plot)

    z = np.zeros((len(x1_plot), len(x2_plot)))
    for i in range(len(x1_plot)):
        for j in range(len(x2_plot)):
            z[i, j] = mapFeature(np.array([x1_plot[i:i+1]]), np.array([x2_plot[j:j+1]])) @ theta

    ax.contour(X1_plot, X2_plot, z, [0], colors='green')

    ax.set_title('Dataset and decision boundary plot with lambda = {}'.format(lmd))
    ax.legend()
    plt.show()

# trainAndPlotDecisionBoundary(X, X_poly, y, theta_init, 0)

# trying different values for the regularization parameter lambda
trainAndPlotDecisionBoundary(X, X_poly, y, theta_init, 0)
trainAndPlotDecisionBoundary(X, X_poly, y, theta_init, 0.01)
trainAndPlotDecisionBoundary(X, X_poly, y, theta_init, 0.5)
trainAndPlotDecisionBoundary(X, X_poly, y, theta_init, 1)
trainAndPlotDecisionBoundary(X, X_poly, y, theta_init, 10)
trainAndPlotDecisionBoundary(X, X_poly, y, theta_init, 100)