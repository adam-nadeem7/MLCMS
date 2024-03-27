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
    # print('x',x)

    if x.size == 2:
        distance_squared = (x[0] - c[0])**2 + (x[1] - c[1])**2

    else:    
        distance_squared = (x[:,0] - c[0])**2 + (x[:,1] - c[1])**2
    return np.exp(-distance_squared / (2 * epsilon**2))

def ls_rbf(x, fx, centers, epsilon):
    """
    Perform least squares radial basis function approximation.

    Parameters:
    - x: Input values
    - fx: Target function values
    - centers: centers of RBF 
    - epsilon: Width parameter of the RBFs
    
    Returns:
    - coefficients: Coefficients obtained from the least squares method
    - residuals: Residuals of the least squares fit
    """
  
    A = np.vstack([rbf(x, c, epsilon) for c in centers]).T

    coefficients, residuals, _, _ = lstsq(A, fx, cond=1e-16)
    return coefficients, residuals

def radial_basis_function_with_coe(x0, coefficients, centers, epsilon):
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
    # print('x0',x0)
    v_approximated = np.zeros_like(x0)
    for i in range(len(coefficients)):

        rbf_value = rbf(x0, centers[i], epsilon)
        if isinstance(rbf_value, np.float64):
            # print('yes')
            v_approximated +=  rbf_value* coefficients[i]
        else:
            v_approximated +=  rbf_value[:, np.newaxis] * coefficients[i]
    return v_approximated
def plot_phase_portrait(U, V, w, title):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    :param U: X component of the vector field.
    :param V: Y component of the vector field.
    :param w: Plotting window size.
    :param title: Title for the plot.
    :return: The plot axis.
    """
    Y, X = np.mgrid[-w:w:60j, -w:w:60j]
    fig, ax0 = plt.subplots(figsize=(10, 10))
    ax0.streamplot(X, Y, U, V, density=[2, 2])
    # ax0.set_title('Streamplot for linear vector field A*x')
    ax0.set_title(title)

    ax0.set_aspect(1)
    plt.savefig(title + '.png', bbox_inches='tight', dpi=100)
    return ax0
def plot_rbf( centers, epsilon, coefficients):
    """
    Plot Radial Basis Function Approximation.

    Parameters:
    - x: Input values
    - fx: Target function values
    - centers: centers of RBF 
    - epsilon: Width parameter of the RBFs
    - coefficients: Coefficients obtained from the least squares method
    """


    # Plot the results
    fig = plt.figure(figsize=(30, 30))

    x0_grid = np.meshgrid(np.linspace(-5, 5, 60), np.linspace(-5, 5, 60))
    x0_generated = np.column_stack([grid.flatten() for grid in x0_grid])
    # print(x0_generated.shape)

    x1_approximated = radial_basis_function_with_coe(x0_generated, coefficients, centers, epsilon)
    plt.quiver(x0_generated[:,0], x0_generated[:,1], x1_approximated[:,0], x1_approximated[:,1], scale=1000, color='blue', width=0.002)
    

    
    plt.legend()
    plt.title(f'Approximation v using RBFs\nEpsilon={epsilon}, L={centers.shape[0]}')
    plt.show()
    plot_phase_portrait(x1_approximated[:,0].reshape(60,60), x1_approximated[:,1].reshape(60,60), 5, f'Phase portrait of nonlinear system\nEpsilon={epsilon}, L={centers.shape[0]}')





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
    fig = plt.figure(figsize=(12, 12))
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