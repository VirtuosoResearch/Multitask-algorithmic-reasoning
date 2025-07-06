# %%
import numpy as np

def generate_data(n_samples=100, n_features=20, noise_std=0.1):
    """
    Generates a dataset for linear regression where:
    - X is sampled from an isotropic Gaussian N(0, I)
    - beta is also sampled from an isotropic Gaussian N(0, I)
    - y = X * beta + noise, where noise ~ N(0, noise_std^2)
    
    Args:
    n_samples (int): Number of samples.
    n_features (int): Number of features (default is 20).
    noise_std (float): Standard deviation of Gaussian noise.
    
    Returns:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    beta (numpy.ndarray): True coefficients of shape (n_features,).
    y (numpy.ndarray): Target vector of shape (n_samples,).
    """
    # Sample X from N(0, I)
    X = np.random.randn(n_samples, n_features)  # Isotropic Gaussian

    # Sample beta from N(0, I)
    beta = np.random.randn(n_features)  # Isotropic Gaussian

    # Generate y with Gaussian noise
    noise = np.random.randn(n_samples) * noise_std
    y = X @ beta + noise  # Linear model with noise

    return X, beta, y

# Example Usage
X, beta, y = generate_data(n_samples=100, n_features=20, noise_std=0.01)

# %%
def ols_solver(X, y):
    """
    Solves the Ordinary Least Squares (OLS) regression problem.
    
    Args:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    y (numpy.ndarray): Target vector of shape (n_samples,).
    
    Returns:
    beta (numpy.ndarray): Estimated coefficients of shape (n_features,).
    """
    # # Add bias (intercept) term to X
    # X = np.column_stack((np.ones(X.shape[0]), X))  # Adding column of ones

    # Compute OLS solution: beta = (X^T X)^(-1) X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

# Solve for beta
estimated_beta = ols_solver(X, y)
print("Estimated Coefficients:", estimated_beta)