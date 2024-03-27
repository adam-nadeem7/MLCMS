# Import necessary libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3  # Used for 3D plotting
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import time
import math
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector
from matplotlib import offsetbox
from sklearn.datasets import load_digits, make_s_curve, make_swiss_roll
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from datafold.utils.plot import plot_pairwise_eigenvector
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator
from plot_functions import *

def preprocess_sr(N, D, noise=0):
    """
    Preprocess data for Swiss Roll with added high-dimensional noise.

    :param N: Number of samples.
    :param D: Target dimensionality (D must be >= 3 for Swiss Roll + noise).
    :param noise: Noise level for data generation.
    :return: Tuple of Point Cloud Manifold, color mapping, and indices for plotting.
    """
    nr_samples = N
    rng = np.random.default_rng(1)

    # Reduce number of points for plotting
    nr_samples_plot = math.floor(N / 10.0)
    idx_plot = rng.permutation(nr_samples)[0:nr_samples_plot]

    # Generate point cloud for Swiss Roll
    X, X_color = make_swiss_roll(nr_samples, random_state=3, noise=noise)

    # Add high-dimensional noise to the Swiss Roll data
    noise = np.random.rand(N, D - 3)  # Swiss Roll is 3D, add noise to reach D dimensions
    X = np.hstack((X, noise))
    
    # Initialize and optimize parameters for the point cloud manifold
    X_pcm = pfold.PCManifold(X)
    X_pcm.optimize_parameters()

    return X_pcm, X_color, idx_plot

def add_noise_to_embeddings(embeddings, noise_level=0):
    """
    Adds Gaussian noise to embeddings.

    :param embeddings: Original embeddings as a NumPy array.
    :param noise_level: Standard deviation of Gaussian noise.
    :return: Embeddings with added Gaussian noise.
    """
    noise = np.random.normal(0, noise_level, embeddings.shape)
    noisy_embeddings = embeddings + noise
    return noisy_embeddings

def preprocess_w2v(word_vectors, N, noise=0, designed_flag=False):
    """
    Preprocess Word2Vec embeddings with optional noise addition.

    :param word_vectors: Original Word2Vec embeddings or similar.
    :param N: Number of embeddings to consider.
    :param noise: Noise level for embeddings.
    :param designed_flag: If True, uses provided word_vectors directly.
    :return: Point Cloud Manifold object after preprocessing.
    """
    if designed_flag:
        word_vector_N = word_vectors
    else:
        word_vector_N = word_vectors.vectors[:N]

    if noise != 0:
        word_vector_N = add_noise_to_embeddings(word_vector_N, noise)
    
    X_pcm = pfold.PCManifold(word_vector_N)
    X_pcm.optimize_parameters()

    return X_pcm

def diffusion(X_pcm, plot=False, n_eigenp=7, intrinsic_dim=2):
    """
    Perform diffusion maps dimensionality reduction.

    :param X_pcm: Point Cloud Manifold object.
    :param plot: If True, returns mapping and diffusion map object for plotting.
    :param n_eigenp: Number of eigenpairs to compute.
    :param intrinsic_dim: Target intrinsic dimensionality.
    :return: Target mapping, optionally returns diffusion map object if plot is True.
    """
    # Adjust subsampling based on data size
    n_subsample = 500 if X_pcm.shape[0] >= 500 else X_pcm.shape[0]
    
    # Initialize and fit the diffusion map
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(
            epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)
        ),
        n_eigenpairs=n_eigenp,
    )
    dmap = dmap.fit(X_pcm)
    
    # Select the target mapping based on local regression selection
    selection = LocalRegressionSelection(
        intrinsic_dim=intrinsic_dim, n_subsample=n_subsample, strategy="dim"
    ).fit(dmap.eigenvectors_)
    target_mapping = selection.transform(dmap.eigenvectors_)

    return (target_mapping, dmap) if plot else target_mapping

def roseland(X_pcm, plot=False, lm=0.25, n_svdtriplet=7, intrinsic_dim=2):
    """
    Perform Roseland dimensionality reduction.

    :param X_pcm: Point Cloud Manifold object.
    :param plot: If True, returns mapping, Roseland object, and SVD vectors for plotting.
    :param lm: Landmark selection parameter.
    :param n_svdtriplet: Number of SVD triplets to compute.
    :param intrinsic_dim: Target intrinsic dimensionality.
    :return: Target mapping, optionally returns Roseland object and SVD vectors if plot is True.
    """
    # Initialize and fit the Roseland model
    rose = dfold.Roseland(
        kernel=pfold.GaussianKernel(
            epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)
        ),
        landmarks=lm,
        n_svdtriplet=n_svdtriplet
    )
    rose = rose.fit(X_pcm)
    
    # Select the target mapping based on local regression selection
    selection = LocalRegressionSelection(
        intrinsic_dim=intrinsic_dim, n_subsample=500, strategy="dim"
    ).fit(rose.svdvec_left_)

    target_mapping = selection.transform(rose.svdvec_left_)

    return (target_mapping, rose, rose.svdvec_left_) if plot else target_mapping

def run_time(X_pcm, method, lm=0.25):
    """
    Calculate the running time of a dimensionality reduction method.

    :param X_pcm: Point Cloud Manifold object.
    :param method: Method to use ('dm' for diffusion maps, 'r' for Roseland).
    :param lm: Landmark selection parameter for Roseland.
    :return: Running time of the method.
    """
    start = time.time()
    if method == 'dm':
        target_mapping = diffusion(X_pcm)
    elif method == 'r':
        target_mapping = roseland(X_pcm, lm=lm)
    end = time.time()

    return end - start

def timer_for_both(X_pcm):
    """
    Compare running times of diffusion maps and Roseland methods.

    :param X_pcm: Point Cloud Manifold object.
    :return: Running times for Roseland and diffusion maps.
    """
    # Calculate running times for both methods
    r_time = run_time(X_pcm, 'r', lm=0.25)
    dm_time = run_time(X_pcm, 'dm')

    return r_time, dm_time
