import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def linear_approximate(data):
    """
    Approximate a linear function to given 2D data using least-squares minimization.

    Parameters:
    - data: 2D numpy array where each row represents a data point with two columns (x, y).

    Returns:
    None (displays the plot of the original data and the fitted line).
    """
    x = data[:,0].T
    y = data[:,1].T
    # print(x)
    # Build designed matrix
    A = np.vstack([x, np.ones_like(x)]).T

    # Use least-squares minimization to fit the line
    result = lstsq(A, y,cond=10e-16)

    # Get the fitting parameters
    m, b = result[0]

    # Plot the original data and the fitted line
    fig = plt.figure(figsize=(8, 8))

    plt.scatter(x, y, label='Original points')
    plt.plot(x, m * x + b, color='red', label='Approximate function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('Approximate linear function')
    plt.show()

    # Print the linear equation
    print(f"Approximate linear function: y = {m:.2f}x + {b:.2f}")


    # Define radial basis function (RBF)
def rbf(x, c, epsilon):
    """
    Radial Basis Function (RBF).

    Parameters:
    - x: Input values
    - c: Center of the RBF
    - epsilon: Width parameter of the RBF

    Returns:
    - RBF value for each input x
    """
    return np.exp(-(x - c)**2 / (2 * epsilon**2))

# Define function for radial basis function approximation
def radial_basis_function(x, coefficients, centers, epsilon):
    """
    Radial Basis Function Approximation.

    Parameters:
    - x: Input values
    - coefficients: Coefficients corresponding to each RBF
    - centers: Centers of the RBFs
    - epsilon: Width parameter of the RBFs

    Returns:
    - Approximated values using radial basis functions
    """
    result = np.zeros_like(x)
    for i in range(len(coefficients)):
        result += coefficients[i] * rbf(x, centers[i], epsilon)
    return result

def ls_rbf(x, fx, l, epsilon):
    """
    Perform least squares radial basis function approximation.

    Parameters:
    - x: Input values
    - fx: Target function values
    - l: Number of RBF centers
    - epsilon: Width parameter of the RBFs
    
    Returns:
    - coefficients: Coefficients obtained from the least squares method
    - residuals: Residuals of the least squares fit
    """
    # Construct design matrix
    centers = np.linspace(min(x), max(x), l)  # Choose l RBF centers
    A = np.vstack([rbf(x, c, epsilon) for c in centers]).T

    # Use least squares method to fit the data
    coefficients, residuals, _, _ = lstsq(A, fx,cond=10e-16)
    return coefficients, residuals

def plot_rbf(x, fx, l, epsilon, coefficients):
    """
    Plot Radial Basis Function Approximation.

    Parameters:
    - x: Input values
    - fx: Target function values
    - l: Number of RBF centers
    - epsilon: Width parameter of the RBFs
    - coefficients: Coefficients obtained from the least squares method
    """

    centers = np.linspace(min(x), max(x), l) 

    # Plot the results
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x, fx, label='original data', alpha=0.5)
    # generated points to plot curve for approximated function
    x_generated = np.linspace(min(x), max(x), 100)
    plt.plot(x_generated, radial_basis_function(x_generated, coefficients, centers, epsilon),
             color='red', label='approximated data', alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('Approximation using RBFs ' + '\n' + f'Epsilon={epsilon}' + f', L={l}')
    plt.show()

def plot_find_para(l_range, e_range, x, fx):
    """
    Plot the residuals for different combinations of L (number of RBF centers) and Epsilon.

    Parameters:
    - l_range: Range of values for L (number of RBF centers), specified as a tuple (start, stop).
    - e_range: Range of values for Epsilon, specified as a tuple (start, stop).
    - x: Input values
    - fx: Target function values

    Returns:
    - None (plots the 3D surface and scatter plot)
    """

    residuals = []  
    s = 10  # Choose a suitable value for the number of points in each dimension

    # Iterate over L and Epsilon ranges
    for i in np.arange(l_range[0], l_range[1], 1):
        for j in np.linspace(e_range[0], e_range[1], s):
            _, residual = ls_rbf(x, fx, i, j)
            
            # Cap the residuals at a maximum value for better visualization
            if residual > 20:
                residual = 20
            
            # Append the combination of L, Epsilon, and the rounded residual to the list
            residuals.append([i, np.round(j, 2), np.round(residual, 3)])

    # Convert the list of residuals to a NumPy array for easier manipulation
    re = np.array(residuals)
    
    # Extract L, Epsilon, and Residual columns
    l = re[:, 0]
    e = re[:, 1]
    r = re[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_surface(l.reshape((s, s)), e.reshape((s, s)), r.reshape((s, s)), cmap='viridis', alpha=0.5)
     # Scatter plot of the data points
    ax.scatter(l, e, r, c='r', marker='o') 

    # Set limits for the z-axis for better visualization

    ax.set_zlim([0, 10]) 

    # Set axis labels
    ax.set_xlabel('L')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Residual')

    # Set plot title
    plt.title('Value of Residual with Different L and Epsilon')

    # Show the plot
    plt.show()