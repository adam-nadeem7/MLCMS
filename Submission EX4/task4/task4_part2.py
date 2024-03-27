import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lorenz system
def lorenz(t, xyz, sigma, beta, rho):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Plot the trajectory in 3D
def plot_trajectory(sol, label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], label=label)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Lorenz System Trajectory')
    ax.legend()
    plt.show()

# Calculate Euclidean distance between two trajectories
def calculate_difference(sol1, sol2):
    return np.linalg.norm(sol1.y - sol2.y, axis=0)

# Plot the difference between trajectories over time
def plot_difference(sol, sol_perturbed):
    difference = calculate_difference(sol, sol_perturbed)
    plt.plot(sol.t, difference, label='||x(t) - x^(t)||')
    plt.xlabel('Time')
    plt.ylabel('Euclidean Distance')
    plt.title('Difference between Trajectories over Time')
    plt.axhline(y=1, color='r', linestyle='--', label='Difference = 1')
    plt.legend()
    plt.show()

# Plot trajectories for given initial conditions and parameters
def plot_lorenz_trajectories(x0, sigma=10, beta=8/3, rho=28, t_span=(0, 1000)):
    # Compute the trajectory for the given initial conditions and parameters
    sol = solve_ivp(lorenz, t_span, x0, args=(sigma, beta, rho), t_eval=np.linspace(0, 1000, 10000))
    
    # Compute trajectory for a slightly perturbed initial condition
    perturbed_x0 = [10 + 1e-8, 10, 10]
    sol_perturbed = solve_ivp(lorenz, t_span, perturbed_x0, args=(sigma, beta, rho), t_eval=np.linspace(0, 1000, 10000))

    # Plot the trajectories
    plot_trajectory(sol, f'x0 = {x0} (ρ = {rho})')
    plot_trajectory(sol_perturbed, f'x0 = {perturbed_x0} (ρ = {rho})')
    
    # Plot the difference between trajectories over time
    plot_difference(sol, sol_perturbed)

# Initial conditions
x0 = [10, 10, 10]

# Plot trajectories for ρ = 28
plot_lorenz_trajectories(x0, rho=28)

# Plot trajectories for ρ = 0.5
plot_lorenz_trajectories(x0, rho=0.5)
