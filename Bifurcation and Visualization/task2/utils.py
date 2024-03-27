import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eig

def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time, values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1] - time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k - 1, :] + step_size * f_ode(yt[k - 1, :])
    return yt, time

def plot_phase_portrait(U, V, w, title):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    :param U: X component of the vector field.
    :param V: Y component of the vector field.
    :param w: Plotting window size.
    :param title: Title for the plot.
    :return: The plot axis.
    """
    # Create a meshgrid for plotting
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]

    # Create a figure with a specified grid layout
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    # Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Streamplot for linear vector field A*x')
    ax0.set_aspect(1)
    ax0.set_title(title)

    ax0.set_aspect(1)
    plt.savefig(title + '.png', bbox_inches='tight', dpi=100)
    return ax0


def plot_portrait_with_traj(f_ode, x, t, y0, w, title):
    """
    Plots the phase portrait along with the trajectory.
    :param f_ode: the right hand side of the ordinary differential equation.
    :param x: Grid as input of f_ode
    :param t: Total time for simulation.
    :param y0: Initial condition.
    :param w: Plotting window size.
    :param title: Title for the plot.
    """
    # Generate a time array
    time = np.linspace(0, t, 100)

    # Solve the ODE using the Euler method
    yt, time = solve_euler(f_ode, y0, time)

    # Plot the phase portrait with streamlines
    ax0 = plot_phase_portrait(f_ode(x)[0], f_ode(x)[1], w, title)

    # Plot the initial point as a green dot
    plt.scatter(y0[0], y0[1], color='green', marker='o')

    # Plot the trajectory over the phase portrait in red
    ax0.plot(yt[:, 0], yt[:, 1], c='red', label='Orbit')
    ax0.legend(loc='upper right')
    ax0.set_title(title)
    ax0.set_aspect(1)
    plt.savefig(title + '_orbit.png', bbox_inches='tight', dpi=100)

def plot_bif_2d(stable, unstable, domain):
    """
    Plot the saddle-node bifurcation diagram.
    :param stable: Function representing the stable state.
    :param unstable: Function representing the unstable state.
    :param domain: Domain of values for parameter α.
    """
    # Create a figure
    fig = plt.figure(figsize=(9, 6))

    # Add a subplot
    ax1 = fig.add_subplot(1, 1, 1)

    # Plot stable and unstable states
    ax1.plot(domain, stable(domain), 'k-', label='stable state', linewidth=3)
    ax1.plot(domain, unstable(domain), 'k--', label='unstable state', linewidth=3)

    # Add legend
    ax1.legend(loc='upper left')

    # Mark the initial point with a red dot
    ax1.plot(domain[0], stable(domain[0]), 'go')

    # Set axis limits
    ax1.axis([-1, 5, -5, 5])

    # Set labels and title
    ax1.set_xlabel('α')
    ax1.set_ylabel('ẋ')
    ax1.set_title('Saddle-node bifurcation')