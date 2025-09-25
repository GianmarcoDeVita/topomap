import statistics

from sklearn.cluster import MiniBatchKMeans, Birch, AffinityPropagation
import hdbscan
import joblib
import keras
import sys
import pickle
from scipy.stats import mode
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from utils.load_data import *


# Use K-Means as clustering method
def kmeans(data_train, data_test, num_clusters):

    # Data for training
    # Reshape training data to make it suitable for K-Means algorithm
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))

    # Data for test
    # Reshape test data to make it suitable for K-Means algorithm
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

    # Train K-Means model
    kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    kmeans.fit(data_train)

    # Compute cluster labels
    cluster_labels_train = kmeans.labels_   # Cluster labels for training set
    cluster_labels_test = kmeans.predict(data_test) # Cluster labels for test set

    return cluster_labels_train, cluster_labels_test, kmeans


# Use BIRCH as clustering method
def birch(data_train, data_test, num_clusters):

    # Data for training
    # Reshape training data to make it suitable for K-Means algorithm
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))

    # Data for test
    # Reshape test data to make it suitable for K-Means algorithm
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

    # Train K-Means model
    birch_model = Birch(n_clusters=num_clusters)
    birch_model.fit(data_train)

    # Compute cluster labels
    cluster_labels_train = birch_model.labels_   # Cluster labels for training set
    cluster_labels_test = birch_model.predict(data_test) # Cluster labels for test set

    # After training, delete the tree structures which are not needed for clusters' prediction
    # (Otherwise it would be impossible to save the model due to tree's recursive structure)
    birch_model.root_ = None
    birch_model.dummy_leaf_ = None

    return cluster_labels_train, cluster_labels_test, birch_model


# Use AffinityPropagation as clustering method
def affinity_propagation(data_train, data_test):

    # Data for training
    # Reshape training data to make it suitable for clustering algorithm
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))

    # Data for test
    # Reshape test data to make it suitable for clustering algorithm
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

    # Train clustering model
    affinity_propagation = AffinityPropagation(copy=False, verbose=True)
    affinity_propagation.fit(data_train)

    # Compute cluster labels
    cluster_labels_train = affinity_propagation.labels_   # Cluster labels for training set
    cluster_labels_test = affinity_propagation.predict(data_test) # Cluster labels for test set

    return cluster_labels_train, cluster_labels_test, affinity_propagation


# Use HDBSCAN as clustering method
def hdbscan_approx(data_train, data_test):

    # Data for training
    # Reshape training data to make it suitable for clustering algorithm
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))

    # Data for test
    # Reshape test data to make it suitable for clustering algorithm
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

    # Train clustering model
    hdbscan_model = hdbscan.HDBSCAN(prediction_data=True)
    hdbscan_model.fit(data_train)

    # Compute cluster labels
    cluster_labels_train = hdbscan_model.labels_   # Cluster labels for training set
    cluster_labels_test, strengths = hdbscan.approximate_predict(hdbscan_model, data_test) # Cluster labels for test set

    # Turn "noise cluster" into an actual cluster
    new_noise_cluster_label = np.max(np.unique(hdbscan_model.labels_)) + 1
    if np.any(cluster_labels_train == -1):
        cluster_labels_train[cluster_labels_train == -1] = new_noise_cluster_label
    if np.any(cluster_labels_test == -1):
        cluster_labels_test[cluster_labels_test == -1] = new_noise_cluster_label

    return cluster_labels_train, cluster_labels_test, hdbscan_model


# Print progress bar for experiments
def progress_bar_experiments(current_experiment, total_experiments):

    # Progress bar logic
    progress = current_experiment + 1
    bar_length = 40

    fraction = progress / total_experiments
    filled_length = int(bar_length * fraction)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    percent = int(fraction * 100)

    print(f'\rExperiment {progress}/{total_experiments} |{bar}| {percent}%', end='')
    sys.stdout.flush()


# Perform clustering to find K clusters and analyze the quality of this clustering
def analyze_num_of_clusters(data_name, data_train, data_test, num_clusters, experiment, test_set_labels, embedding="none", clustering="kmeans"):

    clusters_labels_name = "_".join((str(embedding), str(clustering), str(num_clusters), "experiment", ".".join((str(experiment), "npy"))))
    clusters_train_labels_loc = os.path.join(data_name, "data", "experiments_clusters", ("_".join((data_name, "train", clusters_labels_name))))
    clusters_test_labels_loc = os.path.join(data_name, "data", "experiments_clusters", ("_".join((data_name, "test", clusters_labels_name))))

    if not os.path.exists(clusters_test_labels_loc):

        # Apply clustering
        if clustering == "birch":
            y_train_clust, y_test_clust, model = birch(data_train, data_test, num_clusters)
        else:
            y_train_clust, y_test_clust, model = kmeans(data_train, data_test, num_clusters)

        model_name = os.path.join(data_name, "data", "trained_cluster_models", ".".join(
            ("_".join((data_name, str(embedding), str(clustering), str(num_clusters), "experiment", str(experiment))), "joblib")))
        joblib.dump(model, model_name)

        silhouette = silhouette_score(data_test, y_test_clust)
        db_index = davies_bouldin_score(data_test, y_test_clust)
        ch_index = calinski_harabasz_score(data_test, y_test_clust)

        # Save cluster labels for the training set
        np.save(clusters_train_labels_loc, y_train_clust)

        # Save cluster labels for the test set
        np.save(clusters_test_labels_loc, y_test_clust)

        # Save metrics
        metric_loc = os.path.join(data_name, "data", "metrics", ("silhouette_" + clusters_labels_name))
        save_metric = open(metric_loc, 'wb')
        pickle.dump(silhouette, save_metric)
        save_metric.close()

        metric_loc = os.path.join(data_name, "data", "metrics", ("db_index_" + clusters_labels_name))
        save_metric = open(metric_loc, 'wb')
        pickle.dump(db_index, save_metric)
        save_metric.close()

        metric_loc = os.path.join(data_name, "data", "metrics", ("ch_index_" + clusters_labels_name))
        save_metric = open(metric_loc, 'wb')
        pickle.dump(ch_index, save_metric)
        save_metric.close()

    else:
        y_test_clust = load_pseudolabels(data_name, experiment, embedding, clustering, num_clusters, "test")

        # Load metrics
        metric_loc = os.path.join(data_name, "data", "metrics", ("silhouette_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            silhouette = pickle.load(f)

        metric_loc = os.path.join(data_name, "data", "metrics", ("db_index_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            db_index = pickle.load(f)

        metric_loc = os.path.join(data_name, "data", "metrics", ("ch_index_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            ch_index = pickle.load(f)

    length = len(test_set_labels)
    labels = np.zeros(length)
    for i in range(0, num_clusters):
        mask = y_test_clust == i
        labels[mask] = mode(test_set_labels[mask])[0]
    corresponding_cluster_labels = labels

    cluster_accuracy = accuracy_score(test_set_labels, corresponding_cluster_labels)

    return silhouette, db_index, ch_index, cluster_accuracy


# Find the number of clusters K
def find_K_clusters(data_name, embedding, clustering, data_train_embedded, data_test_embedded, loaded_train_labels,
                    loaded_test_labels, file_out, num_experiments):

    # Set the minimum value of K equal to the number of classes in the classification problem
    minK = len(np.unique(loaded_train_labels))
    # Initialize tested values of K by the minimum K (= num_classes)
    Ks = [minK]
    # Initialize "accuracy" with respect to the "previous K"
    prev_k_acc = 0
    # Set the increment between K(n) and K(n + 1) equal to minK
    increment = minK
    # Initialize the "previous K"
    prev_k = 0
    # Initialize the list of derivatives
    derivatives = [0]

    # Allocate data structures to store conventional clusters evaluation metrics
    experiment_silhouettes = []
    experiment_db_scores = []
    experiment_ch_scores = []

    # Cluster accuracy with respect to the original data labels
    experiment_cluster_accuracy = []

    # Iterate through the values of K (where Ks is a dynamically updated list, hence, it may grow at each iteration)
    for current_k in Ks:
        # Current values of K at this iteration (last value is the current one)
        print("Values of K: ", Ks)

        # In order to account for clustering randomness, we repeat the experiment 10 times
        for exp in range(0, num_experiments):
            keras.backend.clear_session()
            progress_bar_experiments(exp, num_experiments)

            # Conduct analysis for this K
            sil_score, db_score, ch_score, cluster_accuracy = analyze_num_of_clusters(data_name,
                                                                                      data_train_embedded,
                                                                                      data_test_embedded,
                                                                                      current_k, exp,
                                                                                      loaded_test_labels,
                                                                                      embedding, clustering)

            # Append results of quality metrics
            experiment_silhouettes.append(sil_score)
            experiment_db_scores.append(db_score)
            experiment_ch_scores.append(ch_score)
            experiment_cluster_accuracy.append(cluster_accuracy)

        # Go to new line after progress bar is finished
        print('\r' + ' ' * 80 + '\r', end='')  # Overwrite with spaces and return to start
        print("Embedding: ", embedding, " Clustering: ", clustering, " K: ", current_k, " Cluster accuracy: ",
              format(statistics.mean(experiment_cluster_accuracy), ".4f"))

        # Write cluster accuracy and quality metrics report
        file_out.write(embedding + "," + clustering + "," + str(current_k) + ","
                       + str(statistics.mean(experiment_silhouettes)) + ","
                       + str(statistics.mean(experiment_db_scores)) + ","
                       + str(statistics.mean(experiment_ch_scores)) + ","
                       + str(statistics.mean(experiment_cluster_accuracy)) + "\n")
        file_out.flush()

        # Compute the slope of the curve defining the clusters' accuracy
        this_k_acc = statistics.mean(experiment_cluster_accuracy)
        this_der = (this_k_acc - prev_k_acc) / (current_k - prev_k)
        derivatives.append(this_der)

        curr_dev = len(derivatives) - 1
        avg_dev = statistics.mean([derivatives[curr_dev], derivatives[curr_dev - 1]])

        print("This derivative: ", format(this_der, ".4f"))
        print("Average of the last two derivatives: ", format(avg_dev, ".4f"))
        print("-------------------------------------------------------------------------")

        if avg_dev > 0.001:
            # If the average of the last two derivatives is not lower than the tolerance threshold, add a new value of K.
            prev_k_acc = this_k_acc
            prev_k = current_k
            Ks.append(current_k + increment)
        else:
            return current_k


# Perform clustering with automatic assessment of clusters
def find_auto_clusters(data_name, data_train, data_test, experiment, test_set_labels, embedding="none", clustering="affinity_propagation"):

    clusters_labels_name = "_".join((str(embedding), str(clustering), "experiment", ".".join((str(experiment), "npy"))))
    clusters_train_labels_loc = os.path.join(data_name, "data", "experiments_clusters", ("_".join((data_name, "train", clusters_labels_name))))
    clusters_test_labels_loc = os.path.join(data_name, "data", "experiments_clusters", ("_".join((data_name, "test", clusters_labels_name))))

    if not os.path.exists(clusters_test_labels_loc):

        # Apply clustering
        if clustering == "hdbscan":
            y_train_clust, y_test_clust, model = hdbscan_approx(data_train, data_test)
        else:
            y_train_clust, y_test_clust, model = affinity_propagation(data_train, data_test)

        model_name = os.path.join(data_name, "data", "trained_cluster_models", ".".join(
            ("_".join((data_name, str(embedding), str(clustering), "experiment", str(experiment))), "joblib")))
        joblib.dump(model, model_name)

        silhouette = silhouette_score(data_test, y_test_clust)
        db_index = davies_bouldin_score(data_test, y_test_clust)
        ch_index = calinski_harabasz_score(data_test, y_test_clust)

        # Save cluster labels for the training set
        np.save(clusters_train_labels_loc, y_train_clust)

        # Save cluster labels for the test set
        np.save(clusters_test_labels_loc, y_test_clust)

        # Save metrics
        metric_loc = os.path.join(data_name, "data", "metrics", ("silhouette_" + clusters_labels_name))
        save_metric = open(metric_loc, 'wb')
        pickle.dump(silhouette, save_metric)
        save_metric.close()

        metric_loc = os.path.join(data_name, "data", "metrics", ("db_index_" + clusters_labels_name))
        save_metric = open(metric_loc, 'wb')
        pickle.dump(db_index, save_metric)
        save_metric.close()

        metric_loc = os.path.join(data_name, "data", "metrics", ("ch_index_" + clusters_labels_name))
        save_metric = open(metric_loc, 'wb')
        pickle.dump(ch_index, save_metric)
        save_metric.close()

    else:
        y_train_clust = load_pseudolabels_no_k(data_name, experiment, embedding, clustering, "train")
        y_test_clust = load_pseudolabels_no_k(data_name, experiment, embedding, clustering, "test")

        # Load metrics
        metric_loc = os.path.join(data_name, "data", "metrics", ("silhouette_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            silhouette = pickle.load(f)

        metric_loc = os.path.join(data_name, "data", "metrics", ("db_index_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            db_index = pickle.load(f)

        metric_loc = os.path.join(data_name, "data", "metrics", ("ch_index_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            ch_index = pickle.load(f)

    length = len(test_set_labels)
    labels = np.zeros(length)
    num_clusters = len(np.unique(y_train_clust))
    for i in range(0, num_clusters):
        mask = y_test_clust == i
        labels[mask] = mode(test_set_labels[mask])[0]
    corresponding_cluster_labels = labels

    cluster_accuracy = accuracy_score(test_set_labels, corresponding_cluster_labels)

    return silhouette, db_index, ch_index, cluster_accuracy, num_clusters


# Find the number of clusters K
def find_clusters(data_name, embedding, clustering, data_train_embedded, data_test_embedded,
                    loaded_test_labels, file_out, num_experiments):

    # Allocate data structures to store conventional clusters evaluation metrics
    experiment_silhouettes = []
    experiment_db_scores = []
    experiment_ch_scores = []

    # Cluster accuracy with respect to the original data labels
    experiment_cluster_accuracy = []
    experiment_number_of_cluster = []

    current_k = -1

    # In order to account for clustering randomness, we repeat the experiment 10 times
    for exp in range(0, num_experiments):
        keras.backend.clear_session()
        progress_bar_experiments(exp, num_experiments)

        # Conduct analysis for this K
        sil_score, db_score, ch_score, cluster_accuracy, current_k = find_auto_clusters(data_name,
                                                                                  data_train_embedded,
                                                                                  data_test_embedded,
                                                                                  exp,
                                                                                  loaded_test_labels,
                                                                                  embedding, clustering)
        # Record number of clusters into the array
        experiment_number_of_cluster.append(current_k)

        # Append results of quality metrics
        experiment_silhouettes.append(sil_score)
        experiment_db_scores.append(db_score)
        experiment_ch_scores.append(ch_score)
        experiment_cluster_accuracy.append(cluster_accuracy)

    # Go to new line after progress bar is finished
    print('\r' + ' ' * 80 + '\r', end='')  # Overwrite with spaces and return to start
    print("Embedding: ", embedding, " Clustering: ", clustering, " K: ",
          format(statistics.mean(experiment_number_of_cluster)), " Cluster accuracy: ",
          format(statistics.mean(experiment_cluster_accuracy), ".4f"))

    # Write cluster accuracy and quality metrics report
    file_out.write(embedding + "," + clustering + ","
                   + str(format(statistics.mean(experiment_number_of_cluster))) + ","
                   + str(statistics.mean(experiment_silhouettes)) + ","
                   + str(statistics.mean(experiment_db_scores)) + ","
                   + str(statistics.mean(experiment_ch_scores)) + ","
                   + str(statistics.mean(experiment_cluster_accuracy)) + "\n")
    file_out.flush()

    return current_k


# Predict cluster labels on validation set
def predict_clusters_valid_set(data_name, data_valid, experiment, embedding, clustering, num_clusters):

    if clustering == "kmeans" or clustering == "birch":

        # Load clustering model
        model_name = os.path.join(data_name, "data", "trained_cluster_models", ".".join(
            ("_".join((data_name, str(embedding), str(clustering), str(num_clusters), "experiment", str(experiment))),
             "joblib")))
        model = joblib.load(model_name)

        clusters_labels_valid = model.predict(data_valid)

        # Save cluster labels
        clusters_labels_name = "_".join((data_name, "valid", str(embedding), str(clustering), str(num_clusters),
                                         "experiment", ".".join((str(experiment), "npy"))))
    else:

        # Load clustering model
        model_name = os.path.join(data_name, "data", "trained_cluster_models", ".".join(
            ("_".join((data_name, str(embedding), str(clustering), "experiment", str(experiment))), "joblib")))
        model = joblib.load(model_name)

        if clustering == "affinity_propagation":
            clusters_labels_valid = model.predict(data_valid)
        else:
            clusters_labels_valid = hdbscan.approximate_predict(model, data_valid)
            # Turn "noise cluster" into an actual cluster
            new_noise_cluster_label = np.max(np.unique(model.labels_)) + 1
            if np.any(clusters_labels_valid == -1):
                clusters_labels_valid[clusters_labels_valid == -1] = new_noise_cluster_label

        # Save cluster labels
        clusters_labels_name = "_".join((data_name, "valid", str(embedding), str(clustering), "experiment",
                                         ".".join((str(experiment), "npy"))))

    clusters_valid_labels_loc = os.path.join(data_name, "data", "experiments_clusters", clusters_labels_name)

    # Save cluster labels for the validation set
    np.save(clusters_valid_labels_loc, clusters_labels_valid)

    return clusters_labels_valid
