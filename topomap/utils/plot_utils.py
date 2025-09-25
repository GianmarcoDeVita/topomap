import math
import os
from select import select

import umap
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from random import sample

data_folder_name = "plot_data"
data_subfolders_names = ["2d_embedder"]

# Initialize directories for saving plot data
def initialize_plot_directories():

    # Create the directory
    try:
        os.mkdir(data_folder_name)
        # Create subfolders
        for subfolder_name in data_subfolders_names:
            os.mkdir(os.path.join(data_folder_name, subfolder_name))
        print(f"Directory '{data_folder_name}' and sub-directories created successfully!")
    except FileExistsError:
        print(f"Directory '{data_folder_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{data_folder_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

# Import original data
def import_original_data(data_name, set_name):

    # Load original data
    data_path = os.path.join("input", data_name, '_'.join((data_name, set_name, '.'.join(("x", "npy")))))
    loaded_data = np.load(data_path)

    return loaded_data

# Import embedded data
def import_embedded_data(data_name, embedding_config, set_name):

    # Load embedded data
    data_path = os.path.join("data", "embedded_data", '_'.join((data_name, str(embedding_config), "data",
                                                                '.'.join((set_name, "npy")))))
    loaded_data = np.load(data_path)

    return loaded_data


# Import original labels
def import_original_labels(data_name, set_name):

    # Load original data
    data_path = os.path.join("input", data_name, '_'.join((data_name, set_name, '.'.join(("y", "npy")))))
    loaded_labels = np.load(data_path)

    return loaded_labels


# Import cluster labels
def import_cluster_labels(data_name, configuration, set_name, experiment=0):

    # Load cluster labels
    clusters_labels_name = '_'.join((data_name, str(set_name), str(configuration[0]), str(configuration[1]),
                                     str(configuration[2]), "experiment", '.'.join((str(experiment), "npy"))))
    cluster_labels_path = os.path.join("data", "experiments_clusters", clusters_labels_name)
    data_cluster_labels = np.load(cluster_labels_path)

    return data_cluster_labels


# Select class labels
def select_class_labels(data_name, set_name, labels=[]):

    class_labels = import_original_labels(data_name, set_name)

    if len(labels) > 0:

        selected_class_labels = np.where(np.isin(class_labels, labels))[0]

        return selected_class_labels

    else:

        return range(0, len(class_labels))


# Fit model mapping data into 2D space
def fit_to_2d(data_train):

    # Fit an embedding model for UMAP
    embedding_model = umap.UMAP(n_components=2)
    embedded_train_data = embedding_model.fit_transform(data_train)

    return embedding_model


# Map data into 2D space
def transform_to_2d(data, embedding_model):

    # Transform data with the embedder
    embedded_data = embedding_model.transform(data)

    return embedded_data


# Compute cluster-specific medoid
def get_clustered_data_medoid(this_cluster_embedded_data):

    # Reshape the embedded data if it's not already flattened
    if len(this_cluster_embedded_data.shape) > 2:
        this_cluster_embedded_data_flat = this_cluster_embedded_data.reshape(this_cluster_embedded_data.shape[0], -1)
    else:
        this_cluster_embedded_data_flat = this_cluster_embedded_data

    # Compute the pairwise distance matrix
    distance_matrix = squareform(pdist(this_cluster_embedded_data_flat))

    # Compute row sums of distance matrix
    row_sums = np.sum(distance_matrix, axis=1)

    # Get argmin of row_sums (i.e., the cluster-specific index of the medoid)
    argmin_row_sums = np.argmin(row_sums)

    return argmin_row_sums


# Compute medoids for 2D-mapped data
def get_mapped_data_medoids(embedded_2d_data, cluster_labels):

    # Initialize list for medoids by cluster
    cluster_medoids = []
    cluster_medoids_indices = []

    # Get id's of existing clusters
    individual_clusters = np.unique(cluster_labels)

    for cluster in individual_clusters:
        this_cluster_indices = np.where(cluster_labels == cluster)[0]
        this_cluster_embedded_2d_data = embedded_2d_data[this_cluster_indices]

        cluster_medoids.append(this_cluster_embedded_2d_data[get_clustered_data_medoid(this_cluster_embedded_2d_data)])
        cluster_medoids_indices.append(this_cluster_indices[get_clustered_data_medoid(this_cluster_embedded_2d_data)])

    return np.array(cluster_medoids), np.array(cluster_medoids_indices)


# Get selected sample inputs to depict
def get_sample_inputs(data_name, set_name, embedded_2d_data, cluster_labels, selected_class=None):

    # Load original data
    loaded_original_data = import_original_data(data_name, set_name)

    if selected_class is not None:
        loaded_original_data = loaded_original_data[selected_class]

    # Get medoids from 2D mapping of the embedded data
    medoids_for_2d_clusters, medoids_for_2d_clusters_indices = get_mapped_data_medoids(embedded_2d_data, cluster_labels)

    # Get the raw data of selected elements
    selected_sampled_data = loaded_original_data[medoids_for_2d_clusters_indices]

    return medoids_for_2d_clusters_indices, selected_sampled_data


# Plot 2D data
def plot_2d_data(data_name, stacked_embedded_2d_data, data_cluster_labels, num_clusters, experiment, selected_samples=None):

    umap_2d_df = pd.DataFrame(data=stacked_embedded_2d_data, columns=("x", "y", "cluster"))

    # Get color map
    x = np.linspace(0, num_clusters-1, num_clusters)
    cmap = plt.get_cmap('jet', num_clusters)

    # Normalizer
    norm = mpl.colors.Normalize(vmin=0, vmax=num_clusters-1)

    # Creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    colors_cluster_labels = [cmap(i) for i in data_cluster_labels]

    # Topographical map plot
    plt.plot(figsize=(18, 6))
    plt.scatter(umap_2d_df["x"], umap_2d_df["y"], c=colors_cluster_labels, zorder=0)
    if selected_samples:
        # Overlay image on scatter plot
        for img_idx in range(0, len(selected_samples[0])):

            x = umap_2d_df["x"][((selected_samples[0])[img_idx])]
            y = umap_2d_df["y"][((selected_samples[0])[img_idx])]
            w = 1
            h = 1

            plt.imshow((selected_samples[1])[img_idx], cmap='gray', extent=[x - w / 2, x + w / 2, y - h / 2, y + h / 2],
                       origin='upper', zorder=1)

    plt.colorbar(sm, ticks=np.linspace(0, num_clusters-1, np.floor(num_clusters/5).astype(int), dtype=int))
    plt.xlim(np.ceil(np.min(umap_2d_df["x"]) - 1), np.ceil(np.max(umap_2d_df["x"]) + 1))
    plt.ylim(np.ceil(np.min(umap_2d_df["y"]) - 1), np.ceil(np.max(umap_2d_df["y"]) + 1))
    fig_path = os.path.join(data_folder_name, '_'.join((data_name, str(num_clusters), "cluster", '.'.join((
        "_".join(("aggregation", str(experiment))), "pdf")))))
    plt.savefig(fig_path, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()

    return None
