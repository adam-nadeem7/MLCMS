import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib import cm

from sir_model import *
from scipy.integrate import solve_ivp


def simulation_sir(b,random_state):
    t_0 = 0  # Initial time
    t_end = 100  # End time for simulation
    NT = t_end - t_0  # Number of time steps

    # if these error tolerances are set too high, the solution will be qualitatively (!) wrong
    rtol = 1e-8  # Relative tolerance for solver
    atol = 1e-8  # Absolute tolerance for solver

    # SIR model parameters
    beta = 11.5  # Average number of adequate contacts per unit time with infectious individuals
    A = 20  # Recruitment rate of susceptibles (e.g., birth rate)
    d = 0.1  # Natural death rate
    nu = 1  # Disease-induced death rate
    # b = 0.01  # Hospital beds per 10,000 persons (try different values) try to set this to 0.01, 0.020, ..., 0.022, ..., 0.03

    mu0 = 10  # Minimum recovery rate
    mu1=10.45
    print("Reproduction number R0=", R0(beta, d, nu, mu1))
    print('Globally asymptotically stable if beta <= d + nu + mu0. This is', beta <= d + nu + mu0)

    # simulation
    rng = np.random.default_rng(random_state)

    SIM0 = rng.uniform(low=(190, 0, 1), high=(199, 0.1, 8), size=(3,))  # Initial conditions

    time = np.linspace(t_0, t_end, NT)  # Time array for simulation
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='LSODA', rtol=rtol, atol=atol)

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(sol.t, sol.y[0] - 0 * sol.y[0][0], label='1E0*susceptible')
    ax[0].plot(sol.t, 1e3 * sol.y[1] - 0 * sol.y[1][0], label='1E3*infective')
    ax[0].plot(sol.t, 1e1 * sol.y[2] - 0 * sol.y[2][0], label='1E1*removed')
    ax[0].set_xlim([0, 500])
    ax[0].legend()
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")

    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label='recovery rate')
    ax[1].plot(sol.t, 1e2 * sol.y[1], label='1E2*infective')
    ax[1].set_xlim([0, 500])
    ax[1].legend()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    I_h = np.linspace(-0., 0.05, 100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b))
    ax[2].plot(I_h, 0 * I_h, 'r:')
    # ax[2].set_ylim([-0.1,0.05])
    ax[2].set_title("Indicator function h(I)"+'b='+str(b))
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")

    fig.tight_layout()


def sir_trajectory(SIM0_list,show_traj,para,b=0.01,mu1=10.45):
    """
    Visualize SIR (Susceptible-Infective-Removed) model trajectories in 3D space.

    Parameters:
    - SIM0_list (list): List of initial conditions for the simulation.
    - show_traj (bool): If True, plot the trajectory; if False, plot initial and final points.
    - para (str): Parameter for the title ('mu1' or 'b').
    - b (float, optional): Hospital beds per 10,000 persons. Default is 0.01.
    - mu1 (float, optional): Disease-induced death rate. Default is 10.45.
    """
    t_0 = 0  # Initial time
    t_end = 1000000  # End time for simulation
    NT = t_end - t_0  # Number of time steps

    # if these error tolerances are set too high, the solution will be qualitatively (!) wrong
    rtol = 1e-8  # Relative tolerance for solver
    atol = 1e-8  # Absolute tolerance for solver

    # SIR model parameters
    beta = 11.5  # Average number of adequate contacts per unit time with infectious individuals
    A = 20  # Recruitment rate of susceptibles (e.g., birth rate)
    d = 0.1  # Natural death rate
    nu = 1  # Disease-induced death rate
    # b = 0.01  # Hospital beds per 10,000 persons (try different values) try to set this to 0.01, 0.020, ..., 0.022, ..., 0.03

    mu0 = 10  # Minimum recovery rate
    # create a figure
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111,projection="3d")
    time = np.linspace(t_0,t_end,NT)

    for SIM0 in SIM0_list:
      sol = solve_ivp(model, t_span=[time[0],time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='DOP853', rtol=rtol, atol=atol)
      # plot initial point and final point
      ax.scatter(sol.y[0][0], sol.y[1][0], sol.y[2][0],color='green', marker='o',label='Initial point',zorder=2,s=5)
      ax.scatter(sol.y[0][-1], sol.y[1][-1], sol.y[2][-1],color='red', marker='o',label='Final point',zorder=10,s=10)
    # limit the axis



    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")
    # decide parameter for title
    if para ==  'mu1':
       bif_para = mu1
       ax.set_xlim([190, 210])
       ax.set_ylim([0, 0.25])
       ax.set_zlim([0, 8])
       para
    if para ==  'b':
       ax.set_xlim([193, 200])
       ax.set_ylim([0, 0.08])
       ax.set_zlim([0, 7])
       para= 'y0='+str(SIM0_list[0])+' b'
       bif_para = b
      # choose if show trajectory, save fig
    if show_traj == False:
      ax.set_title('Initial and final points for Hopf bifurcation: mu1='+str(mu1)) 
      fig.tight_layout()      
      plt.savefig('Initial and final points for Hopf bifurcation: mu1='+str(mu1)+'.png')
    if show_traj == True:
      ax.scatter(sol.y[0], sol.y[1], sol.y[2],s=3, c='b',zorder=0,alpha=0.1)
      ax.set_title('SIR trajectory for '+para+'='+str(bif_para)) 
      fig.tight_layout()
      plt.savefig('SIR trajectory for backward bifurcation: '+para+'='+str(bif_para)+'.png')
    plt.show()