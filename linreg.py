'''
Roxanne S. Avinante
EE 298 Deep Learning
Assignment #1

Problem:
Implement in python3 and numpy the gradient descent for linear regression of arbitrary polynomial function. 
The program is activated by:

python3 linreg.py 2 4 -1

Where the arguments are the coefficients of the polynomial (ie 2x^2 + 4x - 1) . 
To prove its robustness against noise on input and output, add uniform distribution noise from the unit sphere. 
Expect up to 3rd deg polynomial.

What is the recommended learning rate?
'''

import math
import random
import numpy as np
import sys

def predict_output(feature_matrix, coefficients):
    ''' Returns an array of predictions
    
    inputs - 
        feature_matrix - 2-D array of dimensions data points by features
        coefficients - 1-D array of estimated feature coefficients
        
    output - 1-D array of predictions
    '''
    predictions = np.dot(feature_matrix, coefficients)
    return predictions

def feature_derivative(errors, feature):
    derivative = 2*np.dot(errors, feature)
    return(derivative)

def gradient_descent_regression(H, y, initial_coefficients, eta, epsilon, max_iterations=1000):
    '''  
        H - 2-D array of dimensions data points by features
        y - 1-D array of true output
        initial_coefficients - 1-D array of initial coefficients
        eta - float, the step size eta, the learning rate
        epsilon - float, the tolerance at which the algorithm will terminate
        max_iterations - int, tells the program when to terminate
    
    output - 1-D array of estimated coefficients
    '''
    converged = False
    w = initial_coefficients
    iteration = 0
    while not converged:
        if iteration > max_iterations:
            print('Exceeded max iterations\nCoefficients: ', w)
            print('Learning rate: ',eta)
            return w
        pred = predict_output(H, w)
        residuals = pred-y
        gradient_sum_squares = 0
        for i in range(len(w)):
            partial = feature_derivative(residuals, H[:, i])
            gradient_sum_squares += partial**2
            w[i] = w[i] - eta*partial
        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < epsilon:
            converged = True
        iteration += 1
    print(w)
    return w

import matplotlib.pyplot as plt
import sys

arg_len = len(sys.argv)
coefficient_args = np.array([0 for i in range(arg_len-1)])
for i in range(arg_len-1):
    coefficient_args[i] = sys.argv[i+1]

coefficient_args = coefficient_args.astype(float)

p = np.poly1d(coefficient_args)
print('Model: ', np.poly1d(p))
print('Initial Coefficients: ', coefficient_args)
random.seed(1)
n = 20
pts = [x/5. for x in range(n)]
X = np.array(pts)
#y without noise
y = np.array([p(x) for x in pts])
#y with noise (output)
y = y + [random.gauss(0,1.0/3.0) for i in range(n)]
print('Output (y) with noise: ', y)

feature_matrix = np.zeros(n*len(coefficient_args))
feature_matrix.shape = (n, len(coefficient_args))
if len(coefficient_args) == 1:
	feature_matrix[:,0] = 1
elif len(coefficient_args) == 2:
	feature_matrix[:,0] = X
	feature_matrix[:,1] = 1
elif len(coefficient_args) == 3:
	feature_matrix[:,0] = X**2
	feature_matrix[:,1] = X
	feature_matrix[:,2] = 1
else:
	feature_matrix[:,0] = X**3
	feature_matrix[:,1] = X**2
	feature_matrix[:,2] = X
	feature_matrix[:,3] = 1

print('Feature matrix (X): ', X);
initial_coefficients = np.array([0 for i in range(arg_len-1)])
initial_coefficients = initial_coefficients.astype(float)
print('Initialized Coefficients: ', initial_coefficients)

coef = gradient_descent_regression(feature_matrix, y, initial_coefficients, 0.00006, 1)


# derive y via random x or thru normal distribution using the model with coefficients
# disregard the original coefficients, just get the matrix
# inject noise in the ouput or x
# predict the coefficient using gradient descent