# Multiple Linear Regression in Python

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# Importing the dataset

X = []
Y = []

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1]) 
    Y.append(float(y))

# Converting to numpy arrays

X = np.array(X)
Y = np.array(Y)

# Plotting the data

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# Calculating the weights

W = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, W)

# Computing the R-Squared

Diff1 = Y - Yhat
Diff2 = Y - Y.mean()
RSqu = 1 - Diff1.dot(Diff1) / Diff2.dot(Diff2)
print("R-Squared is: ", RSqu)
