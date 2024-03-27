import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x, iterations):
    """Generate logistic map iterations"""
    results = []
    for _ in range(iterations):
        x = r * x * (1 - x)
        results.append(x)
    return results

def plot_bifurcation(r_values, iterations, title='Logistic Map Iterations for Different r Values', x_label='Iterations'):
    """Plot bifurcation diagram for given r_values"""
    plt.figure(figsize=(10, 8))
    for r in r_values:
        x_values = logistic_map(r, 0.1, iterations)
        plt.plot(x_values, label=f'r={r}')
    set_plot_labels(title, x_label)
    
def set_plot_labels(title, x_label):
    """Set labels and display the plot"""
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('x')
    plt.legend()
    plt.grid(True)
    plt.show()

# Modularized code for bifurcation diagram
def generate_bifurcation(iterations):
    r_values = np.linspace(0, 4, 1000)  # Range of r values from 0 to 4

    # Iterate through different r values and collect steady states
    steady_states = []
    for r in r_values:
        x = 0.1  # Initial value of x
        results = logistic_map(r, x, iterations)
        # Taking only the last 100 iterations to find steady states
        for val in results[-100:]:
            steady_states.append([r, val])

    # Separate the steady states into r and x values
    r_steady = [point[0] for point in steady_states]
    x_steady = [point[1] for point in steady_states]

    # Plotting the bifurcation diagram
    plt.figure(figsize=(10, 6))
    plt.scatter(r_steady, x_steady, s=1, color='blue', marker='.')
    set_plot_labels('Bifurcation Diagram - Logistic Map', 'r')
    

# Range 0 to 2
r_values_0_to_2 = np.linspace(0, 2, 8)
plot_bifurcation(r_values_0_to_2, 101, title='Logistic Map Iterations for Different r Values (0 to 2)', x_label='Iterations')

# Range 2 to 4
r_values_2_to_4 = np.linspace(2, 4, 8)
plot_bifurcation(r_values_2_to_4, 101, title='Logistic Map Iterations for Different r Values (2 to 4)', x_label='Iterations')

# Generating bifurcation diagram
generate_bifurcation(1000)
