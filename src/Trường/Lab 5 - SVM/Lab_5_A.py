# Support Vector Machines
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# loading the dataset into X (inputs) and y (outputs)
mat = loadmat("datasets/lab5data1.mat")
X = mat["X"]
y = mat["y"].reshape(len(X))
print("X.shape:", X.shape, "y.shape:", y.shape)

fig, ax = plt.subplots()
X0, X1 = X[y == 0], X[y == 1]
ax.scatter(X0[:, 0], X0[:, 1], color="red", marker="$0$")
ax.scatter(X1[:, 0], X1[:, 1], color="blue", marker="$1$")
# plt.show()

from sklearn.svm import SVC
def svm_train_and_plot(X, y, C):
    clf = SVC(C=C, kernel='linear').fit(X, y) # training
    theta = np.concatenate([clf.intercept_, clf.coef_[0]]) # the parameters vector theta

    # plotting the dataset and linear decision boundary
    fig, ax = plt.subplots()
    X0, X1 = X[y == 0], X[y == 1]
    ax.scatter(X0[:, 0], X0[:, 1], color="red", marker="$0$")
    ax.scatter(X1[:, 0], X1[:, 1], color="blue", marker="$1$")

    plot_x1 = np.linspace(0, 4)
    plot_x2 = - (theta[0] + theta[1] * plot_x1) / theta[2]
    ax.plot(plot_x1, plot_x2, color="green")

    ax.set_title("SVM Decision Boundary with C = " + str(C))
    plt.show()

#svm_train_and_plot(X, y, 1)
# svm_train_and_plot(X, y, 10)
#svm_train_and_plot(X, y, 100)

# SVM with Gaussian Kernels
def gaussianKernel(u, v, sigma):
    return np.exp(-np.sum((u-v)**2) / 2 * sigma ** 2)

print(gaussianKernel(np.array([1, 2]), np.array([0, 3]), 1.0))

# Example dataset 2
mat = loadmat("datasets/lab5data2.mat")
X = mat["X"]
y = mat["y"].reshape(len(X))
print("X.shape:", X.shape, "y.shape:", y.shape)

fig, ax = plt.subplots()
X0, X1 = X[y == 0], X[y == 1]
ax.scatter(X0[:, 0], X0[:, 1], color="red", marker="$0$")
ax.scatter(X1[:, 0], X1[:, 1], color="blue", marker="$1$")
# plt.show()

from sklearn.svm import SVC
def nonlinear_svm_train_and_plot(X, y, C, sigma):
    print("Please wait. This might take some time (few seconds) ...")

    gamma = 1 / (2 * sigma ** 2)
    clf = SVC(C=C, kernel='rbf', gamma=gamma).fit(X, y) # training

    # plotting the dataset and nonlinear decision boundary
    fig, ax = plt.subplots()
    X0, X1 = X[y == 0], X[y == 1]
    ax.scatter(X0[:, 0], X0[:, 1], color="red", marker="$0$")
    ax.scatter(X1[:, 0], X1[:, 1], color="blue", marker="$1$")

    x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
    x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
    plot_x1, plot_x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.004), np.arange(x2_min, x2_max, 0.004))
    Z = clf.predict(np.c_[plot_x1.ravel(), plot_x2.ravel()])
    Z = Z.reshape(plot_x1.shape)

    ax.contour(plot_x1, plot_x2, Z, colors="green")

    ax.set_title("SVM Decision Boundary with $C = {}, \sigma = {}$".format(C, sigma))
    plt.show()

# nonlinear_svm_train_and_plot(X, y, C=1, sigma=0.01)
# nonlinear_svm_train_and_plot(X, y, C=1, sigma=0.1)
# nonlinear_svm_train_and_plot(X, y, C=1, sigma=0.5)
    
# Example dataset 3
mat = loadmat("datasets/lab5data3.mat")
X = mat["X"]
y = mat["y"].reshape(len(X))
Xval = mat["Xval"]
yval = mat["yval"].reshape(len(Xval))

print("X.shape:", X.shape, "y.shape:", y.shape)
print("Xval.shape:", Xval.shape, "yval.shape:", yval.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
X0, X1 = X[y == 0], X[y == 1]
ax1.scatter(X0[:, 0], X0[:, 1], color="red", marker="$0$")
ax1.scatter(X1[:, 0], X1[:, 1], color="blue", marker="$1$")
ax1.set_title("The training set (X, y)")

Xval0, Xval1 = Xval[yval == 0], Xval[yval == 1]
ax2.scatter(Xval0[:, 0], Xval0[:, 1], color="red", marker="o")
ax2.scatter(Xval1[:, 0], Xval1[:, 1], color="blue", marker="+")
ax2.set_title("The validation set (Xval, yval)")
# plt.show()

def train_nonlinear_svm(X, y, C, sigma):
    gamma = 1 / (2 * sigma ** 2)
    clf = SVC(C=C, kernel='rbf', gamma=gamma).fit(X, y) # training SVM
    return clf

parameters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
clfs, accuracies = [], []
def best():
    for C in parameters:
        for sigma in parameters:
            clf = train_nonlinear_svm(X, y, C, sigma)
            clfs.append((clf, (C, sigma)))

    for clf, parameter in clfs:
        predictions = clf.predict(Xval)
        accuracy = np.mean(predictions == yval) * 100
        accuracies.append(accuracy)

    index = np.argmax(np.array(accuracies)) #index of the highest accuracy in accuracies
    print("Highest Accuracy:", np.mean(clfs[index][0].predict(Xval) == yval) * 100)
    print("C_best=", clfs[index][-1][0], ", sigma_best=", clfs[index][-1][1])
    return clfs[index][-1] #return the best parameter

C_best, sigma_best = best()
nonlinear_svm_train_and_plot(X, y, C_best, sigma_best)