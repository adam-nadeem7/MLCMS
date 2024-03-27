# Import necessary libraries for data manipulation, plotting, and dimensionality reduction
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3  # Used for 3D plotting, noqa: F401 indicates to linters to ignore unused import
import numpy as np
from sklearn.datasets import make_swiss_roll
from datafold.utils.plot import plot_pairwise_eigenvector
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import trustworthiness

import utilities  # Assuming utilities.py contains relevant functions used in plotting and data processing

def plot_ori_swiss_roll(nr_samples_plot=1000):
    """
    Plots the original Swiss Roll manifold with specified number of samples.

    :param nr_samples_plot: Number of samples to plot (default 1000).
    """
    rng = np.random.default_rng(1)
    X, X_color = make_swiss_roll(nr_samples_plot, random_state=3, noise=0)

    # Plotting the Swiss Roll
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X_color, cmap=plt.cm.Spectral)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Point cloud on Swiss Roll manifold")
    ax.view_init(10, 70)

def plot_pair_eigenvector_r(svdvecs, idx_plot, X_color):
    """
    Plots pairwise eigenvectors for Roseland dimensionality reduction.

    :param svdvecs: SVD vectors obtained from Roseland.
    :param idx_plot: Indices of samples to plot.
    :param X_color: Color mapping for the samples.
    """
    plot_pairwise_eigenvector(eigenvectors=svdvecs[idx_plot, :], n=1,
                              fig_params=dict(figsize=[14, 12]),
                              scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]))

def plot_pair_eigenvector_dm(dmap, idx_plot, X_color):
    """
    Plots pairwise eigenvectors for Diffusion Maps dimensionality reduction.

    :param dmap: Diffusion Map object containing eigenvectors.
    :param idx_plot: Indices of samples to plot.
    :param X_color: Color mapping for the samples.
    """
    plot_pairwise_eigenvector(eigenvectors=dmap.eigenvectors_[idx_plot, :], n=1,
                              fig_params=dict(figsize=[15, 15]),
                              scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]))

def plot_selected_vector(target_mapping, idx_plot, X_color):
    """
    Plots the selected vectors after dimensionality reduction.

    :param target_mapping: Mapping of the target dimensionality reduction.
    :param idx_plot: Indices of samples to plot.
    :param X_color: Color mapping for the samples.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(target_mapping[idx_plot, 0], target_mapping[idx_plot, 1], c=X_color[idx_plot], cmap=plt.cm.Spectral)

def plot_vs(changed_para, r_values, dm_values, changed_para_name, y_axis, log_flag=True):
    """
    Plots comparison of Roseland and Diffusion Maps based on a varying parameter.

    :param changed_para: Parameter values that were changed.
    :param r_values: Roseland values for the parameter.
    :param dm_values: Diffusion Maps values for the parameter.
    :param changed_para_name: Name of the parameter that was changed.
    :param y_axis: Y-axis label for the plot.
    :param log_flag: Whether to use logarithmic scale for both axes (default True).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(changed_para, r_values, marker='o', label='Roseland', color='r')
    if len(changed_para) != len(dm_values):
        changed_para = changed_para[:len(dm_values)]
    plt.plot(changed_para, dm_values, marker='o', label='Diffusion Maps', color='b')
    plt.xlabel(changed_para_name)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} vs {changed_para_name} for Roseland vs Diffusion Maps')
    if log_flag:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()

def plot_scatter(target_mapping, words, word_vectors, method, text_flag=True):
    """
    Visualizes reduced word vectors with optional annotations.

    :param target_mapping: Mapping of the reduced word vectors.
    :param words: List of words corresponding to the vectors.
    :param word_vectors: Original word vectors for annotation lookup.
    :param method: Method used for dimensionality reduction.
    :param text_flag: Whether to annotate points with words (default True).
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(target_mapping[:, 0], target_mapping[:, 1])
    plt.title(f'2D data after reducing dimension of Word2Vec with {method}')
    if text_flag:
        texts = []
        for i, word in enumerate(words[:-1]):
            if word in word_vectors:
                text = plt.annotate(word, (target_mapping[i, 0], target_mapping[i, 1]))
                texts.append(text)
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

def plot_trust_vs_num_eigen(word_vectors, min_num, max_num, fixed_dataset_size):
    """
    Plots the trustworthiness score vs number of eigenpairs for a range of eigenvalues.

    :param word_vectors: Word vectors to process.
    :param min_num: Minimum number of eigenpairs to consider.
    :param max_num: Maximum number of eigenpairs to consider.
    :param fixed_dataset_size: Fixed size of the dataset for processing.
    """
    results = []
    for e in range(min_num, max_num):
        for s in range(min_num, max_num):
            if s < e:
                X_pcm = utilities.preprocess_w2v(word_vectors, fixed_dataset_size)
                target_mapping_dm = utilities.diffusion(X_pcm,n_eigenp=e, intrinsic_dim=s)
                trustworthiness_score_dm = trustworthiness(X_pcm, target_mapping_dm, n_neighbors=10)
                results.append({'e': e, 's': s, 'trustworthiness': trustworthiness_score_dm})

    # Process and plot results
    e_to_s_trustworthiness = {result['e']: [] for result in results}
    for result in results:
        e, s, trustworthiness_score = result['e'], result['s'], result['trustworthiness']
        e_to_s_trustworthiness[e].append((s, trustworthiness_score))

    plt.figure(figsize=(10, 8))
    for e, s_trustworthiness_pairs in e_to_s_trustworthiness.items():
        s_values, trustworthiness_scores = zip(*s_trustworthiness_pairs)
        plt.plot(s_values, trustworthiness_scores, label=f'n_e={e}')
    plt.legend()
    plt.title('Trustworthiness Score by s for different number of eigenpair')
    plt.xlabel('Number of selected eigenpair value')
    plt.ylabel('Trustworthiness Score')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
