from scipy import stats as st
from sklearn.manifold import TSNE
from utils.plot_utils import *

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


# Create a plot of the topographical map
def plot_topomap(data_name, configuration, set_type, selected_classes, is_data_image, model_library, experiment=0):

    embedding = configuration[0]
    clustering = configuration[1]
    num_clusters = configuration[2]

    embedded_test_set = np.load(os.path.join(data_name, "data", "embedded_data", '_'.join((data_name, str(embedding),
                                                                                           "data", ".".join((set_type,
                                                                                                             "npy"))))))

    original_test_set = np.load(os.path.join("input", data_name, ".".join(("_".join((data_name, "test", "x")), "npy"))))
    original_labels_test = np.load(os.path.join("input", data_name, ".".join(("_".join((data_name, "test", "y")), "npy"))))

    if is_data_image and model_library == "pytorch":
        # Transpose from (N, C, H, W) -> (N, H, W, C)
        original_test_set = np.transpose(original_test_set, (0, 2, 3, 1))

    if clustering == "kmeans" or clustering == "birch":
        clusters_labels_test = np.load(os.path.join(data_name, "data", "experiments_clusters",
                                                    ".".join(("_".join((data_name, "test", embedding, clustering,
                                                                        str(num_clusters), "experiment",
                                                                        str(experiment))), "npy"))), allow_pickle=True)
    else:
        clusters_labels_test = np.load(os.path.join(data_name, "data", "experiments_clusters",
                                                    ".".join(("_".join((data_name, "test", embedding, clustering,
                                                                        "experiment", str(experiment))), "npy"))),
                                       allow_pickle=True)

    length = len(original_labels_test)
    labels = np.zeros(length)
    for i in np.unique(clusters_labels_test):
        mask = clusters_labels_test == i
        labels[mask] = st.mode(original_labels_test[mask], keepdims=False)[0]
    corresponding_cluster_labels = labels

    # Selected indices
    indices_selected_class_labels = select_class_labels(data_name, "test", selected_classes)
    print("Clusters to plot: ", np.unique(clusters_labels_test[indices_selected_class_labels]))

    # Get input elements of clusters to plot
    element_indices = indices_selected_class_labels

    # Obtain a 2-D embedding of data with TSNE
    model = TSNE(n_components=2, random_state=42, perplexity=35)
    embedded_test_set = embedded_test_set.reshape((len(embedded_test_set), np.prod(embedded_test_set.shape[1:])))

    embedded_test_set = embedded_test_set[element_indices]
    original_test_set = original_test_set[element_indices]
    tsne_data = model.fit_transform(embedded_test_set)

    # Rename clusters by their new order
    new_y_clust_labels = clusters_labels_test[element_indices]

    # Creating a new data frame which help us in plotting the result data
    tsne_class_data = np.vstack((tsne_data.T, new_y_clust_labels)).T
    tsne_df = pd.DataFrame(data=tsne_class_data, columns=("x", "y", "label"))

    # Get color map
    K = len(np.unique(new_y_clust_labels))
    N = K
    cmap = plt.get_cmap('jet', N)

    # Normalizer
    norm = mpl.colors.Normalize(vmin=0, vmax=K-1)

    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    unique_vals, tmp_array_for_colors_ = np.unique(new_y_clust_labels, return_inverse=True)
    colors_new_y_clust_labels = [cmap(i) for i in tmp_array_for_colors_]

    # Create a single large plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Scatter plot
    scatter = ax.scatter(tsne_df["x"], tsne_df["y"], c=colors_new_y_clust_labels)

    # Colorbar
    plt.colorbar(sm, ticks=np.linspace(0, K-1, N), ax=ax)

    if is_data_image:

        # Get medoids from 2D mapping of the embedded data
        medoids_for_2d_clusters, medoids_for_2d_clusters_indices = get_mapped_data_medoids(tsne_data, new_y_clust_labels)

        # Get raw data of selected elements
        selected_sampled_data = original_test_set[medoids_for_2d_clusters_indices]

        # Overlay images on scatter plot
        for img_idx in range(len(medoids_for_2d_clusters_indices)):
            x = tsne_df["x"].iloc[medoids_for_2d_clusters_indices[img_idx]]
            y = tsne_df["y"].iloc[medoids_for_2d_clusters_indices[img_idx]]
            w, h = 8, 8

            ax.imshow(
                selected_sampled_data[img_idx], cmap='gray',
                extent=[x - w / 2, x + w / 2, y - h / 2, y + h / 2],
                origin='upper', zorder=1
            )

    # Set axis limits
    ax.set_xlim(np.ceil(np.min(tsne_df["x"]) - 5), np.ceil(np.max(tsne_df["x"]) + 5))
    ax.set_ylim(np.ceil(np.min(tsne_df["y"]) - 5), np.ceil(np.max(tsne_df["y"]) + 5))

    # Save and show
    topomap_plot_filename = ".".join(("_".join((data_name, "topomap", embedding, clustering, str(num_clusters),
                                                str(experiment))), "pdf"))
    plt.savefig(os.path.join(data_name, "topographical_map", topomap_plot_filename), format='pdf', bbox_inches='tight')
    plt.show()
