# Neural Networks

# dataset and visualization
# from matplotlib import axes
from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.io import loadmat

filename = "datasets/digits-dataset.mat"

mat = loadmat(filename)
X = mat['X']
y = mat['y'].reshape(len(X))
print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

X_pca = PCA(n_components=2).fit_transform(X)

fig, ax = plt.subplots()
X1_pca, X2_pca, X3_pca, X4_pca, X5_pca, X6_pca, X7_pca, X8_pca, X9_pca, X10_pca = X_pca[y==1], X_pca[y==2], X_pca[y==3], X_pca[y==4], X_pca[y==5], X_pca[y==6], X_pca[y==7], X_pca[y==8], X_pca[y==9], X_pca[y==10]
ax.scatter(X1_pca[:,0], X1_pca[:,1], c='red', label='Class 1', marker='1')
ax.scatter(X2_pca[:,0], X2_pca[:,1], c='green', label='Class 2', marker='2')
ax.scatter(X3_pca[:,0], X3_pca[:,1], c='blue', label='Class 3', marker='3')
ax.scatter(X4_pca[:,0], X4_pca[:,1], c='orange', label='Class 4', marker='4')
ax.scatter(X5_pca[:,0], X5_pca[:,1], c='yellow', label='Class 5', marker='$5$')
ax.scatter(X6_pca[:,0], X6_pca[:,1], c='purple', label='Class 6', marker='$6$')
ax.scatter(X7_pca[:,0], X7_pca[:,1], c='pink', label='Class 7', marker='$7$')
ax.scatter(X8_pca[:,0], X8_pca[:,1], c='cyan', label='Class 8', marker='$8$')
ax.scatter(X9_pca[:,0], X9_pca[:,1], c='grey', label='Class 9', marker='$9$')
ax.scatter(X10_pca[:,0], X10_pca[:,1], c='brown', label='Class 10', marker='$10$')
ax.set_xlabel('First principal componant of PCA')
ax.set_ylabel('Second principal componant of PCA')
ax.set_title('Dataset visualization using PCA')
ax.legend()
# plt.show()

def visualize_100_digits(X):
    ids = np.random.choice(len(X), 100, replace=False)
    images = X[ids].reshape(100, 20, 20) # 100 images shaped as 20*20

    print("Plotting digits ... This may take few seconds ...")
    fig, ax = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            img = images[i*10 + j].T
            ax[i][j].imshow(img, cmap='gray')
            ax[i][j].axis('off')

    plt.show()

# visualize_100_digits(X)

# Neural Network
def neural_network_training(X, y, max_iterations=1000, alpha=1e-3, hunits=25):
    a1 = add_row_ones(X.T) # create an input layer of dimension (401*n)
    Y = vectorize_outputs(y) # encode the labels vector y as a matrix Y of dimension (K * n)

    # Generating randomly the initial weights (parameters values)
    Theta1 = np.random.randn(hunits, a1.shape[0]) # Parameters matrix of dimension (hunits * 401)
    Theta2 = np.random.randn(Y.shape[0], hunits + 1) # Parameters matrix of dimension K * (hunits + 1)

    # interative optimization of the parameters
    for itr in range(max_iterations):
        z2, a2, z3, z3 = feedforward(a1, Theta1, Theta2) # feedforward propagation
        DELTA1, DELTA2 = backpropagation(a1, z2, a2, z3, a3, Y, Theta1, Theta2) # backpropagation

        # update the parameters using gradient descent
        Theta1 = Theta1 - alpha * DELTA1
        Theta2 = Theta2 - alpha * DELTA2

    return Theta1, Theta2

# some useful functions
def add_row_ones(A):
    row_ones = np.ones((1, A.shape[1]))
    return np.append(row_ones, A, axis=0)

def vectorize_outputs(y):
    K = len(set(y)) # number of unique classes
    return np.eye(K)[y].T

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

result = add_row_ones(
    np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]])
)
print("\n add_row_ones(...) \n", result)

arr = np.array([2, 0, 2, 3, 1, 1])
result = vectorize_outputs(arr)
print("\n vectorize_outputs(...) corresponding to {} is:\n {}".format(arr, result))

result = sigmoid(np.array([-2, -1, 0, 1, 2]))
print("\n sigmoid(...) \n", result)

result = sigmoid_deriv(np.array([-5, -2, -1, 0, 1, 2, 5]))
print("\n sigmoid_deriv(...) \n", result)

# feedfoward propagation, prediction and cost function
mat = loadmat("datasets/ANNweights.mat")
Theta1 = mat['Theta1']
Theta2 = mat['Theta2']

print("Theta1.shape: {}, Theta2.shape: {}".format(Theta1.shape, Theta2.shape))

# feedforward propagation
def feedforward(a1, Theta1, Theta2):
    z2 = Theta1.dot(a1)
    a2 = sigmoid(z2)
    a2 = add_row_ones(a2)
    z3 = Theta2.dot(a2)
    a3 = sigmoid(z3)
    return z2, a2, z3, a3

a1 = add_row_ones(X.T)
print("Shapes: X: {}, a1: {}, y: {}, Theta1: {}, Theta2: {}".format(X.shape, a1.shape, y.shape, Theta1.shape, Theta2.shape))

ffresults = feedforward(a1, Theta1, Theta2)

# the last element returned from feedforward(...) is a3
a3 = ffresults[-1]
print("a3.shape: {}".format(a3.shape))

# The first column in a3 is the predicted output (as a vector of dim K=10) corresponding of the first data-point
arr = a3[:, 0]
print("*** Output corresponding to the first data-point:\n", arr)
print("*** The corresponding predicted class-label (for the first data-point) is:", np.argmax(arr))

# prdicting class-labels
def predict(X, Theta1, Theta2):
    a1 = add_row_ones(X.T)
    ffresults = feedforward(a1, Theta1, Theta2)
    a3 = ffresults[-1]
    return np.argmax(a3, axis=0)

y_pred = predict(X, Theta1, Theta2)
print("Accuracy:", (y == y_pred).mean() * 100)

# Cost function
Y = vectorize_outputs(y)

def E_unregularized(X, Y, Theta1, Theta2):
    a1 = add_row_ones(X.T)
    ffresults = feedforward(a1, Theta1, Theta2)
    a3 = ffresults[-1]
    n, d = X.shape
    E = -Y*np.log(a3) - (1-Y)*np.log(1-a3)
    return E.sum() / n

cost = E_unregularized(X, Y, Theta1, Theta2)
print("Unregularized cost:", cost)

def E_regularized(X, Y, Theta1, Theta2, lmd):
    n, d = X.shape
    E1 = E_unregularized(X, Y, Theta1, Theta2)
    E2 = (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:]))) * lmd / (2*n)
    return E1 + E2

cost = E_regularized(X, Y, Theta1, Theta2, 1)
print("Regularized cost:", cost)

# Backpropagation
# backpropagation without Regularization
def backpropagation(a1, z2, a2, z3, a3, Y, Theta1, Theta2):
    delta3 = 2 * (a3 - Y)

    result_temp = Theta2.T @ delta3
    result_temp = result_temp[1:] # we ignore the first row
    delta2 = result_temp * sigmoid_deriv(z2)

    n = a1.shape[1]
    DELTA2 = (1/n) * (delta3 @ a2.T)
    DELTA1 = (1/n) * (delta2 @ a1.T)

    return DELTA1, DELTA2

def neural_network_training(X, y, max_iterations=2000, alpha=0.5, hunits=25):
    Y = vectorize_outputs(y)
    a1 = add_row_ones(X.T)

    # Random initialization of the parameters
    Theta1 = np.random.randn(hunits, a1.shape[0])
    Theta2 = np.random.randn(Y.shape[0], hunits + 1)

    for itr in range(max_iterations):
        # Feedforward and backpropagation to obtain the gradients
        z2, a2, z3, a3 = feedforward(a1, Theta1, Theta2)
        D1, D2 = backpropagation(a1, z2, a2, z3, a3, Y, Theta1, Theta2)

        # using gradient descent to update the parameters
        Theta1 = Theta1 - alpha * D1
        Theta2 = Theta2 - alpha * D2

        if itr % 50 == 0:
            print("itr = {}, cost = {}".format(itr, E_unregularized(X, Y, Theta1, Theta2)), end='\r')

    return Theta1, Theta2

Theta1, Theta2 = neural_network_training(X, y)
y_pred = predict(X, Theta1, Theta2)
print("Accuracy:", (y == y_pred).mean() * 100)

# Regularized Neural Network
def neural_network_training(X, y, max_iterations=2000, alpha=0.5, hunits=25, lmd=20):
    Y = vectorize_outputs(y)
    a1 = add_row_ones(X.T)
    n, d = X.shape

    # Random initialization of the parameters
    Theta1 = np.random.randn(hunits, a1.shape[0])
    Theta2 = np.random.randn(Y.shape[0], hunits + 1)

    for itr in range(max_iterations):
        # Feedforward and backpropagation to obtain the gradients
        z2, a2, z3, a3 = feedforward(a1, Theta1, Theta2)
        D1, D2 = backpropagation(a1, z2, a2, z3, a3, Y, Theta1, Theta2)

        D1[:, 1:] += Theta1[:, 1:] * lmd / n
        D2[:, 1:] += Theta2[:, 1:] * lmd / n

        Theta1 = Theta1 - alpha * D1
        Theta2 = Theta2 - alpha * D2

        if itr % 50 == 0:
            print("itr = {}, cost = {}".format(itr, E_regularized(X, Y, Theta1, Theta2, lmd)), end='\r')    

    return Theta1, Theta2

Theta1, Theta2 = neural_network_training(X, y)
y_pred = predict(X, Theta1, Theta2)
print("Accuracy:", (y == y_pred).mean() * 100)

# visualize the hidden layer
def visualize_hidden_layer(Theta1, Theta2):
    # Reshape Theta1 into 25 images of shape 20*20
    images = Theta1[:, 1:].reshape((25, 20, 20))

    # plot with a grid of 5*5 images
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            img = images[i*5 + j].T
            ax[i][j].imshow(img, cmap='gray')
            ax[i][j].axis('off')

    plt.show()

Theta1, Theta2 = neural_network_training(X, y, 2000, 0.5, 25, 20)
visualize_hidden_layer(Theta1, Theta2)

# Trying various values of lambda and max_iterations
Theta1, Theta2 = neural_network_training(X, y, 1000, 0.5, 25, 20)
visualize_hidden_layer(Theta1, Theta2)

Theta1, Theta2 = neural_network_training(X, y, 2000, 0.5, 25, 1)
visualize_hidden_layer(Theta1, Theta2)

Theta1, Theta2 = neural_network_training(X, y, 2000, 0.5, 25, 100)
visualize_hidden_layer(Theta1, Theta2)
