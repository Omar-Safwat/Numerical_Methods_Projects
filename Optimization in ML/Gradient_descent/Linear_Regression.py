# Code build Linear Regression model using custom-built class
# Date: 07/07/2021
# Code by: Omar Safwat

from numpy.lib.function_base import gradient
import Gradient_Descent as gd # Module contains all implementations for all gradient descent algorithms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

class Linear_Regression():

    """Class builds a Linear Regression model"""
    gradient_descent = None # Object of Gradient_Descent Class from library "gd"
    X = None # User input data.
    y = None # Target feature
    theta = None # weights of model
    scaler = None #sklearn object that scales the features

    def __init__(self, X, y):
        # X, and y are 2D Numpy arrays.
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
        """Fit model weights to training data"""
        scale = kwargs.get('standardize', False)

        # Initialize class of the solver the user specified
        # Pop the "standardize" keyward argument and pass the rest to the solver
        # Solver returns the optimum weights for hypothesis function
        if (solver == "batch"):
            self.gradient_descent = gd.Batch_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.batch_GD(**kwargs)
        elif(solver == "mini_batch"):
            self.gradient_descent = gd.Mini_batch_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.mini_batch_GD(**kwargs)
        elif(solver == "stochastic"):
            self.gradient_descent = gd.Stochastic_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.stochastic_GD(**kwargs)
        elif(solver == "momentum"):
            self.gradient_descent = gd.Momentum_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.momentum_GD(**kwargs)
        elif(solver == "Nesterov"):
            self.gradient_descent = gd.Nesterov_GD(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.nesterov_GD(**kwargs)
        elif(solver == "Adagrad"):
            self.gradient_descent = gd.Adagrad(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.adagrad_GD(**kwargs)
        elif(solver == "RMSprop"):
            self.gradient_descent = gd.RMSprop(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.rms_prop_GD(**kwargs)
        elif(solver == "Adam"):
            self.gradient_descent = gd.Adam(self.X, self.y, self.h_theta, self.J_theta, self.J_prime, scale)
            kwargs.pop('standardize', None)
            self.theta = self.gradient_descent.adam_GD(**kwargs)
        else:
            print("Please Check your input to argument \"solver\"")

        if(kwargs.get('standardize', False) == True):
            self.scaler = self.gradient_descent.get_scaler()

        return self.theta
    
    def get_r2_score(self):
        return(self.gradient_descent.get_r2_score())

    def predict(self, X_test):
        """Function outputs model prediction"""
        if(self.scaler is not None):
            x_test_scaled = self.scaler.transform(X_test)
            return self.h_theta(x_test_scaled, self.theta)
        else:
            return self.h_theta(X_test, self.theta)

    def plot_LR_2D(self, show_trials=False, N_trials_to_show= 10):
        """Plot solution"""
        plt.scatter(self.X[: , :], self.y[:])
        upper_limit = np.max(self.X)
        lower_limit = np.min(self.X)

        #Plot best fit 
        x_axis = np.atleast_2d(np.arange(lower_limit - 2, upper_limit + 3))
        # Add a column of ones for theta_0
        if(self.scaler is not None):
            X_plot = np.hstack((np.ones_like(x_axis.T), self.scaler.transform(x_axis.T)))
        else:
            X_plot = np.hstack((np.ones_like(x_axis.T), x_axis.T))

        y = self.h_theta(X_plot, np.reshape(self.theta, (len(self.theta), 1)))
        
        plt.plot(x_axis[0, :], y.T[0, :], color='r')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Linear Regression using Gradient Descent")
        plt.show()
        
        # Step through all trials and plot them on the same figure
        if(show_trials == True):  
            step_size = self.gradient_descent.theta_hist.shape[1] // N_trials_to_show 
            # Minimum step_size should be 1
            step_size = step_size + (step_size < 1) * 1
            plt.scatter(self.X[: , :], self.y[:])

            # History of thetas at different epochs are accessed through the "gradient_decscent" solver object
            for theta in self.gradient_descent.theta_hist[:, ::step_size].T: 
                # TODO use h_theta() to calculate prediction using theta_hist on prepared X data
                theta = np.reshape(theta, (X_plot.shape[1], 1))
                y = self.h_theta(X_plot, theta)
                plt.plot(x_axis[0, :], y.T[0, :], linewidth=0.5)

            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Linear Regression trials using Gradient Descent")
            plt.show()

    def plot_MSE(self):
        """Plot MSE vs Epoch and weights"""
        self.gradient_descent.plot_GD()
    
    def show_summary(self):
        self.gradient_descent.show_summary()
