# Importing the libraries

import numpy as np 
import matplotlib.pyplot as plt 

# Importing the dataset

X = []
Y = []

for line in open('data_poly.csv'):
	x, y = line.split(',')
	x = float(x)
	X.append([1, x, x*x])
	Y.append(float(y))

# Converting to numpy arrays

X = np.array(X)
Y = np.array(Y)

# Plotting the data

plt.scatter(X[:, 1], Y)
plt.show()

# Calculating the weights

W = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, W)

# Plotting the data again
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

# Calculating the R-Squared

Diff1 = Y - Yhat
Diff2 = Y - Y.mean()
RSqu = 1 - Diff1.dot(Diff1) / Diff2.dot(Diff2)
print("The R-Squared is: ", RSqu)