import matplotlib.pyplot as plt
import notebook
import numpy as np

import sys
sys.path.insert(0, 'labutils/')

from labutils.lab2utils import lab2partB1
lab2B1 = lab2partB1()

# loading data from the file into mydata
filename = "datasets/PopulationProfit.csv"
mydata = np.genfromtxt(filename, delimiter=",")

# we have n cities (each line corresponds to one city)
n = len(mydata)

# We take the population values from mydata and reshape it into a matrix X of n
# lines and 1 column (population), i.e. an array of n 1-dimensional feature-vectors
X = mydata[:, 0].reshape(n, 1)

# same with y
y = mydata[:, -1]
print(X.shape)
print(y.shape)

# We get the list of values of the 1st feature (col 0 from X) as follows
population = X[:, 0]

# fig, ax = plt.subplots()
# ax.scatter(population, y, marker='+', color='red')
# ax.set_xlabel("Population of City in 10,000s")
# ax.set_ylabel("Profit in $10,000s")
# plt.show()

def add_all_ones_column(X):
    n, d = X.shape # dimension of the matrix X (n lines, d columns)
    XX = np.ones((n, d + 1)) # new matrix of all ones with one additional column
    XX[:, 1:] = X # set X starting from column 1 (keep only column 0 unchanged)
    return XX

# The following line creates a new data matrix with an additional first column (of ones)
X_new = add_all_ones_column(X)

print(X_new[:10, :])

def h(theta, x):
    return np.dot(x, theta)

def E(theta, X, y):
    n, d = X.shape
    E1 = np.square(h(theta, X) - y).sum()
    EE = E1/(2*n)
    return EE

theta = np.array([0, 0])
print("Initial cost: ", E(theta, X_new, y))

def gradE(theta, X, y):
    n, d = X.shape
    J = h(theta, X) - y
    JJ = np.dot(J, X)
    JJJ = JJ / n
    return JJJ

theta = np.array([0, 0])
print("Test: ", gradE(theta, X_new, y))

alpha = 0.01
theta = np.array([10, -30])
max_iterations = 5000
epsilon = 0.00001
iterations = []
cost = []

for itr in range(max_iterations):
    lab2B1.plot(itr, E, theta, X_new, y)
    prev = E(theta, X_new, y)
    cost.append(prev)
    iterations.append(itr)

    theta = theta - alpha*gradE(theta, X_new, y)

    CONDITION = abs(prev - E(theta, X_new, y)) < epsilon
    if CONDITION:
        break

plt.show()

print("Theta:\n", theta)
new_input = np.array([3.5, 7]).reshape(2, 1)
new_input_one = add_all_ones_column(new_input)
print(new_input_one[0])
prediction1 = new_input_one[0]@theta
prediction2 = new_input_one[1]@theta
print(prediction1, prediction2)