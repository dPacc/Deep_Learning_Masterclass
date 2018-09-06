# Predicting the Systolic Blood Pressure given the "Age" & "Weight"

# Importing the libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Importing the dataset

df = pd.read_excel("mlr02.xls")
X = df.as_matrix()

plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

df['Ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'Ones']]
X2Only = df[['X2', 'Ones']]
X3Only = df[['X3', 'Ones']]

def Rsquared(X, Y):
	W = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
	Yhat = X.dot(W)

	Diff1 = Y - Yhat
	Diff2 = Y - Y.mean()
	Rsqu = 1 - Diff1.dot(Diff1) / Diff2.dot(Diff2)
	return(Rsqu)

print("The RSquared for X2 only is: ", Rsquared(X2Only, Y))
print("The RSquared for X3 only is: ", Rsquared(X3Only, Y))
print("The RSquared for both only is: ", Rsquared(X, Y))