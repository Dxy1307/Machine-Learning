# Regularized Linear Regression
from scipy.io import loadmat

mat = loadmat("datasets/water-level-dataset.mat")

# X and y correspond to a training set that your model will learn on
X = mat['X']
y = mat['y'].reshape(len(X))

# Xval and yval correspond to a cross validation set for determining the regularization parameter
Xval = mat['Xval']
yval = mat['yval'].reshape(len(Xval))

# Xtest and ytest correspond to a test set for evaluating performance. These 
# are unseen examples which your model will not see during training
Xtest = mat['Xtest']
ytest = mat['ytest'].reshape(len(Xtest))

print(X[:3, :])
print(y[:3])
print(Xval[:3, :])
print(yval[:3])
print(Xtest[:3, :])
print(ytest[:3])

# Visualize the data
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.suptitle('Figure')
ax.scatter(X, y, color='red', marker='x')
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')
# plt.show()

# adding a first column of ones to the dataset
import numpy as np

def add_all_ones_column(X):
    n, d = X.shape
    XX = np.ones((n, d + 1))
    XX[:, 1:] = X
    return XX

X_new = add_all_ones_column(X)
Xval_new = add_all_ones_column(Xval)

print(X_new[:3])
print(Xval_new[:3])

# Regularized Linear Regression Cost Function
def E(theta, X, y, lmd):
    n, d = X.shape
    E1 = np.square(X @ theta - y).sum() / (2 * n)
    regularized_para = np.square(theta[1:]).sum() * lmd / (2*n)
    return E1 + regularized_para

theta = np.array([1, 1])
print(E(theta, X_new, y, 1))

# Regularized Linear Regression Gradient
def gradE(theta, X, y, lmd):
    n, d = X.shape
    grad0 = ((np.dot(X, theta) - y) @ X) / n
    gradj = grad0 + (lmd*theta)/n
    grad0[1:] = gradj[1:]
    return grad0

theta = np.array([1, 1])
print(gradE(theta, X_new, y, 1))

# Fitting Linear Regression
import scipy.optimize as op

theta = np.array([0, 0]) # some initial parameters vector
lmd = 0 # we set lambda to zero this time
print("Initial cost:", E(theta, X_new, y, lmd))

res = op.minimize(E, theta, (X_new, y, lmd), 'TNC', gradE)
theta = res.x
print("Final cost:", E(theta, X_new, y, lmd))

# This is a function that plots the original dataset (X, y) and the best fit line:
def plot_linear_fir(X, y, theta):
    fig, ax = plt.subplots()

    # plottin the training data
    ax.scatter(X[:, 0], y, color='red', marker='x')
    ax.set_xlabel('Change in water level (x)')
    ax.set_ylabel('Water flowing out of the dam (y)')

    # plotting the line:
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    plot_x = np.arange(x_min, x_max) # range of values for the x axis
    plot_y = theta[0] + theta[1] * plot_x 
    ax.plot(plot_x, plot_y, color='green', label='Best fit line')
    
    ax.set_title("Plot of the training data and best fit line")
    plt.legend()
    plt.show()

# plot_linear_fir(X, y, theta)

# Bias-variance
def trainLinearReg(X, y, theta_init, lmd):
    res = op.minimize(E, theta_init, (X, y, lmd), 'TNC', gradE)
    return res.x # the best/final parameters vector theta

errs_train, errs_val = [], [] # list to save the training and validation errors
for i in range(2, len(y)): # start from 2 examples at least
    optimal_theta = trainLinearReg(X_new[:i], y[:i], np.array([0,0]), 0)
    errs_train.append(E(optimal_theta, X_new[:i], y[:i], 0))
    errs_val.append(E(optimal_theta, Xval_new, yval, 0))

fig, ax = plt.subplots()

ax.plot(range(2, len(y)), errs_train, label='Training error')
ax.plot(range(2, len(y)), errs_val, label='Validation error')

ax.set_xlabel('Number of training examples')
ax.set_ylabel('Error')
ax.set_title('Linear regression learning curve')
plt.legend()
# plt.show()

# Polynomial Regression
def polyFeatures(X, p):
    n, d = X.shape
    X_poly = np.zeros((n, p))
    for i in range(p):
        X_poly[:, [i]] = np.power(X, i + 1)
    return X_poly

def polyFeatures2(X, p):
    n, d = X.shape
    X_poly = np.zeros((n, p))
    X_poly[:, [0]] = X
    for i in range(1, p):
        X_poly[:, [i]] = X*X_poly[:, [i-1]]
    return X_poly

X_poly = polyFeatures(X, 3)
X_poly2 = polyFeatures2(X, 3)
print(X_poly)

# the regularization parameter lambda
lmd = 50.0

p = 8
X_poly = polyFeatures(X, p) # on the training set
Xval_poly = polyFeatures(Xval, p) # on the validation set
Xtest_poly = polyFeatures(Xtest, p) # on the test set

# mean vector and standard deviation vector
mu = np.mean(X_poly, axis=0)
sigma = np.std(X_poly, axis=0)

# normalizing the training set and validation set using mu and sigma and adding an aditional first column of ones
X_poly_normalized = add_all_ones_column((X_poly - mu) / sigma)
Xval_poly_normalized = add_all_ones_column((Xval_poly - mu) / sigma)

# training to find the optimal parameters vector theta
theta_init = np.zeros(p + 1) # we p features, so we need p + 1 parameters
theta = trainLinearReg(X_poly_normalized, y, theta_init, lmd)

# Plotting the dataset and the polynomial regression curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_xlabel("Change in water level (x)")
ax1.set_ylabel("Water flowing out of the dam (y)")
ax1.set_title(r"Polynomial Regression Fit ($\lambda = {}$)".format(lmd))

ax2.set_xlabel("Number of training examples")
ax2.set_ylabel("Error")
ax2.set_title(r"Polynomial Regression Learning Curve ($\lambda = {}$)".format(lmd))

x_plot = np.linspace(-60, 40, 100)
x_plot_poly = polyFeatures(x_plot.reshape(len(x_plot), 1), p)
x_plot_poly_normalized = add_all_ones_column((x_plot_poly - mu) / sigma)
y_plot = x_plot_poly_normalized @ theta
ax1.plot(x_plot, y_plot, color='green', label='Polynomial fit', linestyle="--")
ax1.scatter(X[:, 0], y, color='red', marker='x', label='Training examples')

# Plotting the learning curves using the training and validation sets
errs_train, errs_val = [], []
for i in range(2, len(y)):
    theta = trainLinearReg(X_poly_normalized[:i], y[:i], theta_init, lmd)
    errs_train.append(E(theta, X_poly_normalized[:i], y[:i], 0))
    errs_val.append(E(theta, Xval_poly_normalized, yval, 0))

ax2.plot(range(2, len(y)), errs_train, label='Training error')
ax2.plot(range(2, len(y)), errs_val, label='Validation error')

ax1.legend()
ax2.legend()
# plt.show()

# adjusting the regularization parameter lambda

# selecting lambda using the validation set
lmd_range = [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
errs_train, errs_val = [], []

for lmd in lmd_range:
    theta = trainLinearReg(X_poly_normalized, y, theta_init, lmd)
    errs_train.append(E(theta, X_poly_normalized, y, 0))
    errs_val.append(E(theta, Xval_poly_normalized, yval, 0))

fig, ax = plt.subplots()
ax.plot(lmd_range, errs_train, label='Training')
ax.plot(lmd_range, errs_val, label='Validation')

ax.set_xlabel('lambda')
ax.set_ylabel('Error')
ax.legend()
# plt.show()

# Computing test set error
Xtest_poly_normalized = add_all_ones_column((Xtest_poly - mu) / sigma)
theta = trainLinearReg(X_poly_normalized, y, theta_init, 3)
err_test = E(theta, Xtest_poly_normalized, ytest, 0)
print("err_test =", err_test)

# Optional: Performing a 10-fold-cross-validation
Xall = np.append(X, Xval, axis=0)
yall = np.append(y, yval, axis=0)
