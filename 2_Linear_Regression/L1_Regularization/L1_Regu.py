# L1 Regularization

from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

# Uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5)*10

# True weights - only the first 3 dimensions of X affect Y
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

# Generate Y - add noise with variance 0.5
Y = X.dot(true_w) + np.random.randn(N)*0.5

# Perform gradient descent to find w
costs = [] # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.001
l1 = 10.0 # Also try 5.0, 2.0, 1.0, 0.1 - what effect does it have on w?
for t in range(500):
  # Update w
  Yhat = X.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

  # Find and store the cost
  mse = delta.dot(delta) / N
  costs.append(mse)

# Plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# Plot our w vs true w
plt.plot(true_w, label='True w')
plt.plot(w, label='W_Map')
plt.legend()
plt.show()