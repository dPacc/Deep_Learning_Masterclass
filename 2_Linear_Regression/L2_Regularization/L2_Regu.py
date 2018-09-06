# L2 Regularization/ Ridge-Regression

import numpy as np 
import matplotlib.pyplot as plt 

# Generating some random data

N = 50

X = np.linspace(0, 10, N)
Y = 0.5 * X + np.random.randn(N)

# Manually making some outliers

Y[-1] = +30
Y[-2] = +30

# Plotting the data

plt.scatter(X, Y)
plt.show()

# Adding the Bias term

X = np.vstack([np.ones(N), X]).T 

# Calculate the Maximum-Liklihood solution

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:, 1], Y)
plt.scatter(X[:, 1], Yhat_ml)
plt.show()

# L2 Regularization Solution

l2 = 1000
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml, label = 'Maximum-Liklihood')
plt.plot(X[:, 1], Yhat_map, label = 'Map')
plt.legend()
plt.show()