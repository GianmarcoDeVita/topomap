import numpy as np
import umap
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from utils.set_up_directories import *


# Compute the number of dimensions in the embedding space
def get_num_components(training_data):

    # Initialize number of dimensions in the embedding space
    num_components = 0

    # Find how many components express at least 90% of the variance
    pca = decomposition.PCA()
    pca.n_components = min(len(training_data), len(training_data[0]))
    pca_data = pca.fit_transform(training_data)
    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)
    for i in range(0, len(cum_var_explained)):
        if cum_var_explained[i] < 0.9:
            num_components += 1
    del training_data, pca_data, pca

    return num_components


# Fit an embedder on the input training set
def compute_embedding(embedding, x_train_original, y_train_original):

    # Reshape training set in order to make it suitable to be processed by embedding algorithms
    data_train = np.asarray(x_train_original)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))

    # Get number of dimensions in the embedding space
    num_components = get_num_components(data_train)

    # Create a thread-safe stop signal
    stop_event = threading.Event()

    # Start the animation
    t = threading.Thread(target=walking_mouse, args=(stop_event,))
    t.start()

    # Apply embedding (PCA)
    if embedding == "pca":

        # Fit an embedding model for PCA
        embedding_model = decomposition.PCA()
        embedding_model.n_components = num_components
        embedded_train_data = embedding_model.fit_transform(data_train)

    # Apply embedding (UMap)
    if embedding == "umap":

        # Fit an embedding model for UMAP
        embedding_model = umap.UMAP(n_components=num_components)
        embedded_train_data = embedding_model.fit_transform(data_train)

    # Apply embedding (LDA):
    if embedding == "lda":

        # Fit an embedding model for LDA
        num_components = min(num_components, len(np.unique(y_train_original)) - 1)
        embedding_model = LinearDiscriminantAnalysis(n_components=num_components)
        embedded_train_data = embedding_model.fit_transform(data_train, y=y_train_original)

    # Apply embedding (Isomap):
    if embedding == "isomap":

        # Fit an embedding model for Isomap
        embedding_model = Isomap(n_components=num_components)
        embedded_train_data = embedding_model.fit_transform(data_train)

    # Stop the animation
    stop_event.set()
    t.join()

    # Clear the animation line
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()

    return embedded_train_data, embedding_model, num_components


# Apply embedding on the input data set
def embed_data(x_original_set, embedding_model):

    # Reshape set in order to make it suitable to be processed by embedding algorithms
    data = np.asarray(x_original_set)
    data = data.reshape((len(data), np.prod(data.shape[1:])))

    # Transform data with the embedder
    embedded_data = embedding_model.transform(data)

    return embedded_data