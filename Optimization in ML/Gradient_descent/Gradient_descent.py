# Code implements custom-built gradient descent classes
# Date: 07/07/2021
# Code by: Omar Safwat

# A Gradient Descent interface 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class Gradient_descent():
    """Interface is used as template for gradient descent algorithms"""

    # Interface Attributes
    theta_hist = None # History of weights
    cost_hist = None # History of cost function
    h_pred = None # hyposthesis/prediction
    epoch = None
    alpha = None # Learning rate
    X = None
    y_actual = None
    H_func = None # Hypothesis function
    cost_func = None # cost function
    J_prime = None # Gradient of loss function
    grad_hist = None
    MAX_EPOCHS = None
    n_points = None
    stop_criteria = None

    def __init__(self, X, y, h_func, cost, jacob):
        self.X = X
        self.y_actual = y
        self.H_func = h_func
        self.cost_func = cost
        self.J_prime = jacob
    
    def initialize(self, alpha, guess, batch_size):
        """Initialize first epoch"""
        if guess == 0:
            self.theta_hist = np.zeros((self.X.shape[1], 1)) # Initialize weight vector
        else:
            self.theta_hist = guess

        self.h_pred = self.H_func(self.X[:batch_size, :], self.theta_hist)
        self.cost_hist = self.cost_func(self.h_pred, self.y_actual[:batch_size])
        self.grad_hist = self.J_prime(self.h_pred, self.y_actual[:batch_size], self.X[:batch_size, :], batch_size)
        self.n_points = batch_size
        self.epoch = 0
        self.alpha = alpha

    def update_weights_GD(self, idx_1, idx_2, theta):
        """Function updates weights along the direction of steepest descent and stores results"""
        self.epoch += 1
        # Update weight parameters
        grad_new = self.J_prime(self.h_pred, self.y_actual[idx_1 : idx_2], self.X[idx_1 : idx_2, :], self.n_points)
        self.grad_hist = np.hstack((self.grad_hist, grad_new))
        theta = theta - self.alpha * grad_new
        self.theta_hist = np.hstack((self.theta_hist, theta))
        self.h_pred = self.H_func(self.X[idx_1 : idx_2, :], theta)
        self.cost_hist = np.append(self.cost_hist, self.cost_func(self.h_pred, self.y_actual[idx_1 : idx_2]))

    def shuffle_data(self):
        """Shuffle data points"""
        shuffled_order = np.random.permutation(len(self.y_actual))
        self.X = self.X[shuffled_order, :]
        self.y_actual = np.atleast_2d(self.y_actual[shuffled_order])

    def is_converged(self):
        """Returns boolean if stop criteria is reached"""
        # Computes the difference between two last cost functions
        return abs(self.cost_hist[-2] - self.cost_hist[-1]) < self.stop_criteria

    def show_summary(self):
        """Prints a brief summary after stop criteria is reached"""
        print("Solver summary:")
        print("=" * len("Solver summary:"))
        print("Number of iterations: ", self.epoch)
        print("MSE: ", self.cost_hist[-1])
        print("Stop criteria was reached first: ", self.is_converged())
        print("Model Training accuracy: ", self.get_r2_score())

    def plot_GD(self):
        """Function plots MSE against epochs, and first two parameters"""
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        # Plot Error vs Epoch
        axs[0].plot(np.arange(0, self.epoch + 1), self.cost_hist, linestyle='-', marker='.', markerfacecolor='r')
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("MSE")

        # Plot Error vs theta_0
        axs[1].plot(self.theta_hist[0, :], self.cost_hist, linestyle='-', marker='.', markerfacecolor='r')
        axs[1].set_xlabel("$\\theta_0$")
        axs[1].set_ylabel("MSE")
        # Plot Error vs theta_1
        axs[2].plot(self.theta_hist[1, :], self.cost_hist, linestyle='-', marker='.', markerfacecolor='r')
        axs[2].set_xlabel("$\\theta_1$")
        axs[2].set_ylabel("MSE")

        #fig.tight_layout()
        fig.suptitle("Convergance at $\\alpha =$ " + str(self.alpha), fontsize=16)
        plt.show()

    def get_r2_score(self):
        theta = np.atleast_2d(self.theta_hist[:, -1]).T
        self.h_pred = self.H_func(self.X, theta)
        return(r2_score(self.y_actual, self.h_pred))


class Batch_GD(Gradient_descent):
    """Implementation of Gradient Descent for vanilla gradient descent"""

    def batch_GD(self, guess=0, alpha=0.001, max_epochs=1e3, stop_criteria=1e-3):
        """Optimize weights using Vanilla Gradient Descent"""
        # Initialize first epoch
        idx_2 = len(self.y_actual)
        self.initialize(alpha, guess, idx_2)
        self.MAX_EPOCHS = max_epochs
        self.stop_criteria = stop_criteria

        converged = False
        while (self.epoch < self.MAX_EPOCHS and  converged == False):
            # Update weight values
            self.update_weights_GD(0, idx_2, np.atleast_2d(self.theta_hist[:, -1]).T)
            converged = self.is_converged()
        
        return self.theta_hist[:, -1]

class Mini_batch_GD(Gradient_descent):
    """Mini batch implementation of Gradient Descent"""
    
    n_batches = None

    def mini_batch_GD(self, guess=0, alpha=0.001, n_batches=8, max_epochs=1e3, stop_criteria=1e-3, seed=None):
        """Optimize weights using Mini_batch Gradient Descent"""
        # Randomly shuffle the dataset's order
        if (seed is not None):
            np.random.seed(seed)
        self.n_batches = n_batches
        self.shuffle_data()
        self.MAX_EPOCHS = max_epochs
        self.stop_criteria = stop_criteria

        batch_size = len(self.y_actual) // n_batches
        
        # Initialize first epoch
        self.initialize(alpha, guess, batch_size)

        converged = False
        while (self.epoch < self.MAX_EPOCHS and converged == False):
            for i in range(n_batches):
                idx_1 = i * batch_size
                idx_2 = idx_1 + batch_size
                self.update_weights_GD(idx_1, idx_2, np.atleast_2d(self.theta_hist[:, -1]).T)
            self.shuffle_data()
            converged = self.is_converged()
        
        return self.theta_hist[:, -1]

class Stochastic_GD(Gradient_descent):
    """Stochastic implementation of Gradient Descent"""

    def stochastic_GD(self, guess=0, alpha=0.001, max_epochs=1e3, stop_criteria=1e-3):
        """Optimizes Solution using Stochastic Gradient Descent"""

        # Initialize first epoch
        self.initialize(alpha, guess, 1)
        self.MAX_EPOCHS = max_epochs
        self.stop_criteria = stop_criteria

        converged = False
        while (self.epoch < self.MAX_EPOCHS and converged == False):
            for i in range(len(self.y_actual)):
                self.update_weights_GD(i, i + 1, np.atleast_2d(self.theta_hist[:, -1]).T)
            converged = self.is_converged()

        return self.theta_hist[:, -1]

class Momentum_GD(Gradient_descent):
    """Momentum implementation of Gradient Descent"""

    mu = None # Momentum
    gamma = None

    def initialize(self, alpha, guess, gamma, batch_size):
        """Initialize first epoch"""
        super().initialize(alpha, guess, batch_size)
        self.gamma = gamma
        self.mu = self.gamma * self.J_prime(self.h_pred, self.y_actual[:batch_size], self.X[:batch_size, :], batch_size)
    
    # Override
    def update_weights_GD(self, idx_1, idx_2, theta):
        """Function updates weights along the direction of steepest descent and stores results"""
        self.epoch += 1
        # Update weight parameters
        grad_new = self.J_prime(self.h_pred, self.y_actual[idx_1 : idx_2], self.X[idx_1 : idx_2, :], self.n_points)
        self.grad_hist = np.hstack((self.grad_hist, grad_new))
        self.mu = self.gamma * self.mu + self.alpha * grad_new
        theta = theta - self.mu
        # Record in history
        self.theta_hist = np.hstack((self.theta_hist, theta))
        self.h_pred = self.H_func(self.X[idx_1 : idx_2, :], theta)
        self.cost_hist = np.append(self.cost_hist, self.cost_func(self.h_pred, self.y_actual[idx_1 : idx_2]))

    def momentum_GD(self, guess=0, alpha=0.001, gamma=0.8, max_epochs=1e3, stop_criteria=1e-3):
        """Batch Gradient Descent with momentum"""
        self.MAX_EPOCHS = max_epochs
        self.stop_criteria = stop_criteria
        # Initialize first epoch
        idx_2 = len(self.y_actual)
        self.initialize(alpha, guess, gamma, idx_2)

        converged = False
        while (self.epoch < self.MAX_EPOCHS and  converged == False):
            self.update_weights_GD(0, idx_2, np.atleast_2d(self.theta_hist[:, -1]).T)
            converged = self.is_converged()

        return self.theta_hist[:, -1]

class Nesterov_GD(Momentum_GD):
    """Nesterov implementation of Gradient Descent"""

    def update_weights_GD(self, idx_1, idx_2, theta):
        """Function updates weights along the direction of steepest descent and stores results"""
        self.epoch += 1
       
        # Projecting along the gradient to "look ahead"
        grad_proj = self.J_prime(self.h_pred, self.y_actual[idx_1 : idx_2], self.X[idx_1 : idx_2, :], self.n_points)
        mu_proj = self.gamma * self.mu + self.alpha * grad_proj
        theta_proj = theta - mu_proj
        h_proj = self.H_func(self.X[idx_1 : idx_2, :], theta_proj)

        # Use the gradient of the projected weight to update current weight 
        grad_new = self.J_prime(h_proj, self.y_actual[idx_1 : idx_2], self.X[idx_1 : idx_2, :], self.n_points)
        self.grad_hist = np.hstack((self.grad_hist, grad_new))
        self.mu = self.gamma * self.mu + self.alpha * grad_new
        theta = theta - self.mu
      
        # Record in history
        self.theta_hist = np.hstack((self.theta_hist, theta))
        self.h_pred = self.H_func(self.X[idx_1 : idx_2, :], theta)
        self.cost_hist = np.append(self.cost_hist, self.cost_func(self.h_pred, self.y_actual[idx_1 : idx_2]))
    
    def nesterov_GD(self, guess=0, alpha=0.001, gamma=0.8, max_epochs=1e3, stop_criteria=1e-3):
        """Momentum Gradient descent with look ahead correction using Nesterov's algorithm"""
        self.MAX_EPOCHS = max_epochs
        self.stop_criteria = stop_criteria
        # Initialize first epoch
        idx_2 = len(self.y_actual)
        self.initialize(alpha, guess, gamma, idx_2)

        converged = False
        while (self.epoch < self.MAX_EPOCHS and  converged == False):
            self.update_weights_GD(0, idx_2, np.atleast_2d(self.theta_hist[:, -1]).T)
            converged = self.is_converged()

        return self.theta_hist[:, -1]
        