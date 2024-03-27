# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot
import plotly.graph_objects as go

# Set the backend for matplotlib to 'agg'
matplotlib.use('agg')
# Switch the backend to 'Agg' for compatibility
matplotlib.pyplot.switch_backend('Agg')

def file_df_to_count_df(df,
                        ID_SUSCEPTIBLE=1,
                        ID_INFECTED=0,
                        ID_REMOVED=2):
    """
    Converts the file DataFrame to a group count DataFrame that can be plotted.
    The ID_SUSCEPTIBLE and ID_INFECTED specify which ids the groups have in the Vadere processor file.
    """
    # Get unique pedestrian IDs and simulation times from the DataFrame
    pedestrian_ids = df['pedestrianId'].unique()
    sim_times = df['simTime'].unique()

    # Create an empty DataFrame for group counts
    group_counts = pd.DataFrame(columns=['simTime', 'group-s', 'group-i', 'group-r'])
    group_counts['simTime'] = sim_times
    group_counts['group-s'] = 0
    group_counts['group-i'] = 0
    group_counts['group-r'] = 0

    # Loop through pedestrian IDs to calculate group counts at each simulation time
    for pid in pedestrian_ids:
        simtime_group = df[df['pedestrianId'] == pid][['simTime', 'groupId-PID5']].values
        current_state = ID_SUSCEPTIBLE

        # Update group counts based on state transitions
        group_counts.loc[group_counts['simTime'] >= 0, 'group-s'] += 1
        for (st, g) in simtime_group:
            if g != current_state and g == ID_INFECTED and current_state == ID_SUSCEPTIBLE:
                current_state = g
                group_counts.loc[group_counts['simTime'] > st, 'group-s'] -= 1
                group_counts.loc[group_counts['simTime'] > st, 'group-i'] += 1
            elif g != current_state and g == ID_REMOVED and current_state == ID_INFECTED:
                current_state = g
                group_counts.loc[group_counts['simTime'] > st, 'group-i'] -= 1
                group_counts.loc[group_counts['simTime'] > st, 'group-r'] += 1
                break

    return group_counts


def create_folder_data_scatter(folder):
    """
    Create scatter plot from folder data.
    :param folder:
    :return:
    """
    # Construct the file path for SIRinformation.csv in the given folder
    file_path = os.path.join(folder, "SIRinformation.csv")

    # Return None if the file does not exist
    if not os.path.exists(file_path):
        return None

    # Read data from the CSV file into a DataFrame
    data = pd.read_csv(file_path, delimiter=" ")

    print(data)

    # Define IDs for susceptible, infected, and removed groups
    ID_SUSCEPTIBLE = 1
    ID_INFECTED = 0
    ID_REMOVED = 2

    # Convert file DataFrame to group count DataFrame
    group_counts = file_df_to_count_df(data, ID_INFECTED=ID_INFECTED, ID_SUSCEPTIBLE=ID_SUSCEPTIBLE, ID_REMOVED=ID_REMOVED)

    # Plot group counts
    group_counts.plot()

    # Create scatter plots for susceptible, infected, and recovered groups
    scatter_s = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-s'],
                           name='Susceptible',
                           mode='lines')
    scatter_i = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-i'],
                           name='Infected',
                           mode='lines')
    scatter_r = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-r'],
                           name='Recovered',
                           mode='lines')

    # Return scatter plots and group counts
    return [scatter_s, scatter_i, scatter_r], group_counts
