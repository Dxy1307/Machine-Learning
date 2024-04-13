import matplotlib.pyplot as plt
from scipy.io import loadmat
mat = loadmat("datasets/simpleDataset.mat")
X = mat["X"]
print("X.shape:", X.shape)
fig, ax = plt.subplots()
fig.suptitle("Your Title Here")
ax.scatter(X[:, 0], X[:, 1])
plt.show()