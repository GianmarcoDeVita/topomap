import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING logs
import argparse
import warnings
import re
import gc
from utils.apply_embedding import *
from utils.apply_clustering import *
from utils.dnn_cluster_evaluation import *
from utils.map_selection import *
from utils.reports import *
from utils.plot_topomap import *
import tensorflow as tf
import logging


def generate_map(data_name, model_library, clear_dirs=False, backup_dirs=False, selected_classes=[], is_data_image=False, is_problem_regression=False):

    if not os.path.exists(data_name):
        initialize_directories(data_name)
    if clear_dirs:
        clear_directories(data_name)
    if backup_dirs:
        backup_directories(data_name)

    # Experimental configurations
    embeddings = ["pca", "umap", "lda", "isomap"]
    clusterings = ["kmeans", "birch", "hdbscan"]
    experiments = 10

    # Check whether the input is valid and whether there is a validation set
    input_is_valid, there_is_validation_set = what_to_load_data(data_name)

    if not input_is_valid:
        return -1

    # Ignore warning messages in output
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Initialize reports
    file_out, file_sel, file_pw_acc, file_ncomp = generate_reports(data_name)

    # Load data into routine
    if there_is_validation_set:
        # If there is a validation set, load the training, the validation and the test sets and the respective labels.
        (loaded_train_set, loaded_train_labels, loaded_test_set, loaded_test_labels, loaded_valid_set,
         loaded_valid_labels) = load_data(data_name, there_is_validation_set)
        if is_problem_regression:
            loaded_train_labels, loaded_test_labels, loaded_valid_labels = get_label_buckets(loaded_train_labels, loaded_test_labels, loaded_valid_labels)
    else:
        # If there is not a validation set, load the training and the test sets and the respective labels.
        loaded_train_set, loaded_train_labels, loaded_test_set, loaded_test_labels = load_data(data_name,
                                                                                               there_is_validation_set)
        if is_problem_regression:
            loaded_train_labels, loaded_test_labels = get_label_buckets(loaded_train_labels, loaded_test_labels)

    # Iterate through the different embeddings
    for embedding in embeddings:
        # Check if there exists an embedded training set
        if not os.path.exists(
                os.path.join(data_name, "data", "embedded_data", '_'.join((data_name, str(embedding),
                                                                           "data", "train.npy")))):

            print(f"Compute embedding {embedding} on data.")

            # If not, apply embedding to the training data
            data_train_embedded, embedding_model_, num_components_embedding = compute_embedding(embedding,
                                                                                                loaded_train_set,
                                                                                                loaded_train_labels)
            # Save embedded training data
            np.save(os.path.join(data_name, "data", "embedded_data", '_'.join((data_name, str(embedding),
                                                                               "data", "train.npy"))),
                    data_train_embedded)

            # Save embedding model
            pickle.dump(embedding_model_, open(
                os.path.join(data_name, "data", "trained_embedding_models", '_'.join((data_name,
                                                                                      '.'.join((str(embedding),
                                                                                                'pkl'))))),
                "wb"))

            # Save number of components in the embedded data (dimensionality)
            np.save(
                os.path.join(data_name, "data", "embedded_data",
                             '_'.join((data_name, "number_of_components", '.'.join((str(embedding), 'npy'))))),
                num_components_embedding)

            # Apply embedding to test data
            data_test_embedded = embed_data(loaded_test_set, embedding_model_)
            # Save embedded test data
            np.save(os.path.join(data_name, "data", "embedded_data", '_'.join((data_name, str(embedding),
                                                                               "data", "test.npy"))),
                    data_test_embedded)

            if there_is_validation_set:
                # Apply embedding to validation data
                data_valid_embedded = embed_data(loaded_valid_set, embedding_model_)
                # Save embedded validation data
                np.save(os.path.join(data_name, "data", "embedded_data", '_'.join((data_name, str(embedding),
                                                                                   "data", "valid.npy"))),
                        data_valid_embedded)
        else:
            print(f"Load data for {data_name} dataset embedded with {embedding}.")
            # If an embedded training data already exists, load the embedded training, validationa and test sets
            data_train_embedded = np.load(
                os.path.join(data_name, "data", "embedded_data", '_'.join((data_name, str(embedding),
                                                                           "data", "train.npy"))))
            data_test_embedded = np.load(
                os.path.join(data_name, "data", "embedded_data", '_'.join((data_name, str(embedding),
                                                                           "data", "test.npy"))))
            num_components_embedding = np.load(
                os.path.join(data_name, "data", "embedded_data",
                             '_'.join((data_name, "number_of_components", '.'.join((str(embedding), 'npy'))))))

            if there_is_validation_set:
                # Load the validation set, if it does exist
                data_valid_embedded = np.load(os.path.join(data_name, "data", "embedded_data", '_'.join((data_name,
                                                                                                         str(embedding),
                                                                                                         "data",
                                                                                                         "valid.npy"))))

        # Write in report the number of components for this embedding
        file_ncomp.write(embedding + "," + str(num_components_embedding) + "\n")
        file_ncomp.flush()

        # Iterate through clustering algorithms
        for clustering in clusterings:
            print(f"Perform clustering with {clustering}.")
            if clustering == "kmeans" or clustering == "birch":
                current_k = find_K_clusters(data_name, embedding, clustering, data_train_embedded, data_test_embedded,
                                    loaded_train_labels, loaded_test_labels, file_out, experiments)
            else:
                current_k = find_clusters(data_name, embedding, clustering, data_train_embedded, data_test_embedded,
                                          loaded_test_labels, file_out, experiments)

            # Allocate two variables for storing information on the min_weighted_pw_accuracy and the weighted_pw_accuracy.
            experiment_pairwise_cluster_min_weighted_accuracy = []
            experiment_pairwise_cluster_weighted_accuracy = []

            for experiment in range(0, experiments):
                gc.collect()
                progress_bar_experiments(experiment, experiments)

                if clustering == "kmeans" or clustering == "birch":

                    y_train_clusters = load_pseudolabels(data_name, experiment, embedding, clustering, current_k, "train")

                    y_test_clusters = load_pseudolabels(data_name, experiment, embedding, clustering, current_k, "test")

                else:

                    y_train_clusters = load_pseudolabels_no_k(data_name, experiment, embedding, clustering, "train")

                    y_test_clusters = load_pseudolabels_no_k(data_name, experiment, embedding, clustering, "test")

                if there_is_validation_set:

                    if clustering == "kmeans" or clustering == "birch":

                        # Save cluster labels
                        clusters_labels_name = "_".join((str(embedding), str(clustering), str(current_k),
                                                         "experiment", ".".join((str(experiment), "npy"))))
                        clusters_valid_labels_loc = os.path.join(data_name, "data", "experiments_clusters",
                                                                 ("_".join(("valid", clusters_labels_name))))

                        if not os.path.exists(clusters_valid_labels_loc):

                            # If labels for the validation set do not currently exist, compute them
                            y_valid_clusters = predict_clusters_valid_set(data_name, data_valid_embedded,
                                                                          experiment, embedding, clustering,
                                                                          current_k)
                        else:

                            y_valid_clusters = load_pseudolabels(data_name, experiment, embedding, clustering,
                                                                 current_k, "valid")

                    else:

                        # Save cluster labels
                        clusters_labels_name = "_".join((str(embedding), str(clustering),
                                                         "experiment", ".".join((str(experiment), "npy"))))
                        clusters_valid_labels_loc = os.path.join(data_name, "data", "experiments_clusters",
                                                                 ("_".join(("valid", clusters_labels_name))))

                        if not os.path.exists(clusters_valid_labels_loc):

                            # If labels for the validation set do not currently exist, compute them
                            y_valid_clusters = predict_clusters_valid_set(data_name, data_valid_embedded,
                                                                          experiment, embedding, clustering,
                                                                          current_k)
                        else:

                            y_valid_clusters = load_pseudolabels_no_k(data_name, experiment, embedding, clustering, "valid")

                    min_weighted_accuracy_pair, mean_weighted_accuracy_pair = dnn_eval_with_val(data_name,
                                                                                                loaded_train_set,
                                                                                                y_train_clusters,
                                                                                                loaded_valid_set,
                                                                                                y_valid_clusters,
                                                                                                loaded_test_set,
                                                                                                y_test_clusters,
                                                                                                current_k,
                                                                                                embedding,
                                                                                                clustering,
                                                                                                experiment,
                                                                                                file_pw_acc,
                                                                                                model_library)
                else:
                    min_weighted_accuracy_pair, mean_weighted_accuracy_pair = dnn_eval(data_name,
                                                                                       loaded_train_set,
                                                                                       y_train_clusters,
                                                                                       loaded_test_set,
                                                                                       y_test_clusters,
                                                                                       current_k,
                                                                                       embedding,
                                                                                       clustering,
                                                                                       experiment,
                                                                                       file_pw_acc,
                                                                                       model_library)

                experiment_pairwise_cluster_min_weighted_accuracy.append(min_weighted_accuracy_pair)
                experiment_pairwise_cluster_weighted_accuracy.append(mean_weighted_accuracy_pair)

            # Go to new line after progress bar is finished
            print('\r' + ' ' * 80 + '\r', end='')  # Overwrite with spaces and return to start

            file_sel.write(embedding + "," + clustering + "," + str(current_k) + ","
                           + str(statistics.mean(experiment_pairwise_cluster_min_weighted_accuracy)) + ","
                           + str(statistics.mean(experiment_pairwise_cluster_weighted_accuracy)) + "\n")
            file_sel.flush()

            # Save metrics data
            metric_loc = os.path.join(data_name, "data", "experiments",
                                      (data_name + "_" + str(embedding) + "_" + str(clustering) + "_" +
                                       "experiment_pairwise_cluster_min_weighted_accuracy"))
            save_acc = open(metric_loc, 'wb')
            pickle.dump(experiment_pairwise_cluster_min_weighted_accuracy, save_acc)
            save_acc.close()

            metric_loc = os.path.join(data_name, "data", "experiments",
                                      (data_name + "_" + str(embedding) + "_" + str(clustering) + "_" +
                                       "experiment_pairwise_cluster_weighted_accuracy"))
            save_acc = open(metric_loc, 'wb')
            pickle.dump(experiment_pairwise_cluster_weighted_accuracy, save_acc)
            save_acc.close()

    file_ncomp.close()
    file_pw_acc.close()
    file_out.close()

    # Export Selected Topographical Map
    save_topographical_map(data_name, experiments, there_is_validation_set)

    # Plot Topographical Map for the test set
    sel_embedding_algorithm, sel_cluster_algorithm, selected_k = select_map(data_name)
    plot_topomap(data_name, (sel_embedding_algorithm, sel_cluster_algorithm, selected_k), "test", selected_classes, is_data_image, model_library)


def main():
    parser = argparse.ArgumentParser(description="TopoMap: Topographical Mapper of the Test Input Space")

    # Positional args
    parser.add_argument('input_dataset_name', type=str, help='Input dataset name')

    # Optional args
    parser.add_argument('-p', '--package', type=str, default="keras",
                        help='Python model\'s package: either "keras" or "pytorch"')
    parser.add_argument('-b', '--backup', type=bool, default=False,
                        help='Create a compressed backup copy of the existing data for this dataset."')
    parser.add_argument('-c', '--clear', type=bool, default=False,
                        help='Perform a clean run of the code for this dataset. WARNING: You\'ll lose all your data!"')
    parser.add_argument('-s', '--select', type=str, default="[]", help='Select classes to be shown in the TopoMap plot. Default: all classes."')
    parser.add_argument('-i', '--imgplot', type=bool, default=False, help='Plot clusters medoids. WARNING: Only works with image datasets."')
    parser.add_argument('-r', '--regression', type=bool, default=False, help='Input dataset pertains a regression problem."')

    args = parser.parse_args()

    if len(args.input_dataset_name) < 1:
        ValueError("Please insert a dataset name!")
    cleaned_input_sel_classes = re.sub(r'[^0-9,]', '', args.select)
    if cleaned_input_sel_classes.strip() == '':
        selected_class_labels = []
    else:
        selected_class_labels = [int(x) for x in cleaned_input_sel_classes.split(',')]
    generate_map(args.input_dataset_name, args.package, args.clear, args.backup, selected_class_labels, args.imgplot, args.regression)


if __name__ == "__main__":
    welcome()
    main()
