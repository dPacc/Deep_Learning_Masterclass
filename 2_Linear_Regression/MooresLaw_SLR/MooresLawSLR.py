# Moores Law

import re
import numpy as np 
import matplotlib.pyplot as plt 

# Loading the data

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
	r = line.split('\t')
	x = int(non_decimal.sub('', r[2].split('[')[0]))
	y = int(non_decimal.sub('', r[1].split('[')[0]))
	X.append(x)
	Y.append(y)

# Converting it into numpy arrays

X = np.array(X)
Y = np.array(Y)

# Plotting the data

plt.scatter(X, Y)
plt.show()

# Since we know that the according to Moore's Law, the number of transistor count doubles, the problem is not linear
# So we need to take the log, to make it Linear to apply Linear Regression


Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# Now the solution for Linear Regression

Denom = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / Denom
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / Denom

Yhat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# Computing the R-Squared

Diff1 = Y - Yhat
Diff2 = Y - Y.mean()
RSqu = 1 - ( Diff1.dot(Diff1) / Diff2.dot(Diff2) )

print("a: ", a, "b: ", b)
print("The R-Squared is: ", RSqu)


# What we really want to find is how long it would take the transistor time to double
# We do some mathematical derivations to find that

# log(tc) = a * year + b
# tc = exp(b) * exp(a * year)
# 2 * tc = 2 * exp(b) * exp(a * year)
#        = exp(b) * exp(a * year + ln(2))
# exp(b) * exp(a * year2) = exp(b) * exp(a * year1 + ln2)\
# a * year2 = a * year1 + ln2
# year2 = year1 + ln2/a 

print("Time to double: ", np.log(2)/a, "Years")