import os
import numpy as np


def valid_input(train_flag, test_flag, valid_flag, train_labels_flag, test_labels_flag, valid_labels_flag):

    if not (train_flag and train_labels_flag):
        if not train_flag:
            print("Training data is missing.")
        if not train_labels_flag:
            print("Training labels are missing.")
        return False
    if not (test_flag and test_labels_flag):
        if not test_flag:
            print("Test data is missing.")
        if not test_labels_flag:
            print("Test labels are missing.")
        return False
    if valid_flag or valid_labels_flag:
        # We expect a validation set at this point
        if not (valid_flag and valid_labels_flag):
            if not valid_flag:
                print("Valid data is missing.")
            if not valid_labels_flag:
                print("Valid labels are missing.")
            return False

    return True


def what_to_load_data(data):

    # Get list of inputs in data folder
    input_files = os.listdir(os.path.join("input", data))

    train_ok, test_ok, valid_ok = [False, False, False]
    train_labels_ok, test_labels_ok, valid_labels_ok = [False, False, False]
    # Check whether there are proper training and test sets files and, potentially, a validation set file.
    for input_name in input_files:

        this_input = '_'.join(input_name.split('_')[-2:])

        if this_input == "train_x.npy":
            train_ok = True
        if this_input == "train_y.npy":
            train_labels_ok = True
        if this_input == "valid_x.npy":
            valid_ok = True
        if this_input == "valid_y.npy":
            valid_labels_ok = True
        if this_input == "test_x.npy":
            test_ok = True
        if this_input == "test_y.npy":
            test_labels_ok = True

    input_is_valid = valid_input(train_ok, test_ok, valid_ok, train_labels_ok, test_labels_ok, valid_labels_ok)

    return input_is_valid, valid_ok


def load_data(data, validation_set):

    # Load input sets
    loaded_train_set = np.load(os.path.join("input", data, '_'.join((data, "train", "x.npy"))))
    loaded_train_labels = np.load(os.path.join("input", data, '_'.join((data, "train", "y.npy"))))
    loaded_test_set = np.load(os.path.join("input", data, '_'.join((data, "test", "x.npy"))))
    loaded_test_labels = np.load(os.path.join("input", data, '_'.join((data, "test", "y.npy"))))

    if validation_set:
        loaded_valid_set = np.load(os.path.join("input", data, '_'.join((data, "valid", "x.npy"))))
        loaded_valid_labels = np.load(os.path.join("input", data, '_'.join((data, "valid", "y.npy"))))

        return loaded_train_set, loaded_train_labels, loaded_test_set, loaded_test_labels, loaded_valid_set, loaded_valid_labels

    return loaded_train_set, loaded_train_labels, loaded_test_set, loaded_test_labels


def load_pseudolabels(data_name, experiment, embedding, clustering, k_, type_):

    # Load cluster labels from previous runs
    clusters_labels_name = '_'.join((data_name, str(type_), str(embedding), str(clustering), str(k_),
                                           "experiment", ".".join((str(experiment), "npy"))))
    clusters_labels_loc = os.path.join(data_name, "data", "experiments_clusters", clusters_labels_name)
    y_cluster_labels_test = np.load(clusters_labels_loc)

    return y_cluster_labels_test


def load_pseudolabels_no_k(data_name, experiment, embedding, clustering, type_):

    # Load cluster labels from previous runs
    clusters_labels_name = '_'.join((data_name, str(type_), str(embedding), str(clustering),
                                           "experiment", ".".join((str(experiment), "npy"))))
    clusters_labels_loc = os.path.join(data_name, "data", "experiments_clusters", clusters_labels_name)
    y_cluster_labels_test = np.load(clusters_labels_loc)

    return y_cluster_labels_test


# Bucket continuous/regression labels
def get_label_buckets(train_labels, test_labels, valid_labels=np.array([])):

    if len(valid_labels) > 0:
        y_labels = np.concatenate((train_labels, valid_labels, test_labels))
    else:
        y_labels = np.concatenate((train_labels, test_labels))

    std_array = np.std(train_labels)

    index = 0
    bucket1 = []
    bucket2 = []
    bucket3 = []

    for element in y_labels:
        if element < -std_array:
            bucket1.append(element)
        elif -std_array <= element <= std_array:
            bucket2.append(element)
        elif element > std_array:
            bucket3.append(element)

    buckets = [bucket1, bucket2, bucket3]

    unique_label_list = (0, 1, 2)
    unique_inverse = [-1] * len(y_labels)
    unique_count = [-1] * len(unique_label_list)
    bucket_ind = 0
    for bucket in buckets:
        unique_count[bucket_ind] = len(bucket)
        for element in bucket:
            indices = np.argwhere(y_labels == element)
            for index in indices:
                unique_inverse[index[0]] = bucket_ind

        bucket_ind = bucket_ind + 1

    if len(valid_labels) > 0:
        # Use np.split with the indices where to split
        split_points = [len(train_labels), len(train_labels) + len(valid_labels)]

        split_arrays = np.split(np.asarray(unique_inverse), split_points)
        train_buckets, valid_buckets, test_buckets = split_arrays
        return train_buckets, valid_buckets, test_buckets
    else:
        # Use np.split with the indices where to split
        split_points = [len(train_labels)]

        split_arrays = np.split(np.asarray(unique_inverse), split_points)
        train_buckets, test_buckets = split_arrays
        return train_buckets, test_buckets




