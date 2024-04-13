import matplotlib.pyplot as plt
import notebook
import numpy as np

import sys
sys.path.insert(0, 'labutils/')

from labutils.lab2utils import lab2partA1, lab2partA2
lab2A1, lab2A2 = lab2partA1(), lab2partA2()

# minimizing a function of one parameter with gradient descent
def F(a):
    return (a + 5) ** 2

def dF(a):
    return 2 * (a + 5)

alpha = 0.1 # the learning rate of gradient descent
a = 7 # the initial value of a (any initial value is ok)
max_iterations = 100 # maximum number of iterations to perform
epsilon = 0.0001 # some small number to test for convergence (i.e. to stop if F(a) does not decrease too much)

for itr in range(max_iterations):
    lab2A1.plot(itr, F, a) # this plots an animation
    prev = F(a) # save the value of F(a)

    a -= alpha*dF(a)

    CONDITION = abs(prev - F(a)) < epsilon
    if CONDITION:
        break

# plt.show()

# Minimizing a function of two parameters with gradient descent
def F(a, b):
    return 5 + a**2 + 1.5 * b**2 + a*b

def dFa(a, b):
    return 2*a + b

def dFb(a, b):
    return 3*b + a

alpha = 0.1
a, b = 80, 90
max_iterations = 100
epsilon = 0.0001

for itr in range(max_iterations):
    lab2A2.plot(itr, F, a, b)
    prev = F(a, b)

    a -= alpha*dFa(a, b)
    b -= alpha*dFb(a, b)

    CONDITION = abs(prev - F(a, b)) < epsilon
    if CONDITION:
        break

# Minimizing a function of multiple parameters with gradient descent
def F(theta):
    return np.square(theta).sum()

def gradF(theta):
    return 2*theta;

alpha = 0.1
theta = np.array([80, 90, -20])
max_iterations = 100
epsilon = 0.000001
theta_history = []
iterations = []

for itr in range(max_iterations):
    prev = F(theta)
    theta_history.append(prev)
    iterations.append(itr)
    print("iteration = {}, theta = {}, F(theta) = {}".format(itr, theta, prev))

    theta = theta - alpha*gradF(theta)

    CONDITION = abs(prev - F(theta)) < epsilon
    if CONDITION:
        break

fig, ax = plt.subplots()
ax.plot(iterations, theta_history)
ax.set_xlabel("Number of iterators")
ax.set_ylabel("F(theta)")
plt.show()