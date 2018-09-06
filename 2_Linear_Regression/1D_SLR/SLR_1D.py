# Simple Linear Regression (1-Dimensional)

import numpy as np
import matplotlib.pyplot as plt

# Loading the data
X = []
Y = []

for line in open('data_1d.csv'):
	x, y = line.split(',')
	X.append(float(x))
	Y.append(float(y))

# Converting them into numpy arrays
X = np.array(X)
Y = np.array(Y)

# Visualizing the data
plt.scatter(X, Y)
plt.show()

# Applying the equations to find 'a' and 'b'
Denom = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / Denom
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / Denom

# Calculating the predicted Y
Yhat = a * X + b

# Plotting 
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# Calculating the R-Squared
Diff1 = Y - Yhat
Diff2 = Y - Y.mean()
RSqu = 1 - Diff1.dot(Diff1) / Diff2.dot(Diff2)
print("The R-Squared Error is: ", RSqu)