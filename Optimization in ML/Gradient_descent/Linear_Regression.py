# Code build Linear Regression model using custom-built class
# Date: 07/07/2021
# Code by: Omar Safwat

import Gradient_descent as gd # Module contains all implementations for all gradient descent algorithms
import numpy as np
import matplotlib.pyplot as plt

class Linear_regression():

    """Class builds a Linear Regression model"""
    gradient_descent = None # Object of Gradient_Descent Class from library "gd"
    X = None
    y = None
    theta = None

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def h_theta(self, X, theta):
        """Computes h_func function"""
        return (X @ theta)

    def J_theta(self, h, y):
        """Computes Cost function"""
        return  0.5 * np.mean((h - y) ** 2)
    
    def J_prime(self, h, y, X, n_points):
        """Computes Jacobian vector of Cost function"""
        return np.atleast_2d(((h - y).T @ X)).T / n_points
    
    def fit(self, solver="batch", **kwargs):
        if (solver == "batch"):
            self.gradient_descent = gd.Batch_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime)
            self.theta = self.gradient_descent.batch_GD(**kwargs)
        elif(solver == "mini_batch"):
            self.gradient_descent = gd.Mini_batch_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime)
            self.theta = self.gradient_descent.mini_batch_GD(**kwargs)
        elif(solver == "stochastic"):
            self.gradient_descent = gd.Stochastic_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime)
            self.theta = self.gradient_descent.stochastic_GD(**kwargs)
        elif(solver == "momentum"):
            self.gradient_descent = gd.Momentum_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime)
            self.theta = self.gradient_descent.momentum_GD(**kwargs)
        elif(solver == "Nesterov"):
            self.gradient_descent = gd.Nesterov_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime)
            self.theta = self.gradient_descent.nesterov_GD(**kwargs)
        else:
            print("Please Check your input to argument \"solver\"")

        return self.theta
    
    def get_r2_score(self):
        return(self.gradient_descent.get_r2_score())

    def plot_LR_2D(self, show_trials=False, N_trials_to_show= 10):
        """Plot solution"""
        plt.scatter(self.X[: ,1:], self.y[:])
        upper_limit = np.max(self.X)
        lower_limit = np.min(self.X)

        #Plot best fit 
        theta = self.theta
        plt.plot(np.arange(lower_limit - 2, upper_limit + 3), theta[0] + theta[1] * np.arange(lower_limit - 2, upper_limit + 3), color='r')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Linear Regression using Gradient Descent")
        plt.show()
        
        if(show_trials == True):  
            step_size = self.gradient_descent.theta_hist.shape[1] // N_trials_to_show 
            step_size = step_size + (step_size < 1) 
            plt.scatter(self.X[: ,1:], self.y[:])
            upper_limit = np.max(self.X)
            lower_limit = np.min(self.X)
            for theta in self.gradient_descent.theta_hist[:, ::step_size].T: 
                plt.plot(np.arange(lower_limit - 2, upper_limit + 3), theta[0] + theta[1] * np.arange(lower_limit - 2, upper_limit + 3), linewidth=0.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Linear Regression trials using Gradient Descent")
            plt.show()

    def plot_MSE(self):
        """Plot MSE vs Epoch and weights"""
        self.gradient_descent.plot_GD()
    
    def show_summary(self):
        self.gradient_descent.show_summary()
