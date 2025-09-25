import os
import pandas as pd
import numpy as np
from utils.load_data import load_pseudolabels


def select_map(data, criterion="default"):

    # Load results obtained from TopoMap to get topographical configuration
    df = pd.read_csv(os.path.join(data, "reports",
                                  "_".join(("topomap_report_configurations_pairwise", ".".join((data, "csv"))))), sep=',')

    if criterion == "maxMean":
        selected_configuration = df.loc[df["PairWiseMeanAccuracy"].idxmax()]
    else:
        selected_configuration = df.loc[df["PairWiseMinAccuracy"].idxmax()]

    # Retrieve information on selected configuration
    embedding_algorithm = selected_configuration['Embedding']
    cluster_algorithm = selected_configuration['Clustering']
    selected_k = selected_configuration['K']

    return embedding_algorithm, cluster_algorithm, selected_k


def save_topographical_map(data, num_experiments=10, validation_set="False", criterion="default"):

    # Select Topographical Map
    embedding_algorithm, cluster_algorithm, selected_k = select_map(data, criterion)

    for experiment in range(0, num_experiments):

        labels_filename_suffix = "_".join(("topomap", embedding_algorithm, cluster_algorithm, str(selected_k),
                                           "experiment", ".".join((str(experiment), "npy"))))

        topomap_labels_folder = os.path.join(data, "topographical_map", "cluster_labels")
        if not os.path.exists(topomap_labels_folder):
            os.mkdir(topomap_labels_folder)

        # Load map of the training set
        topographical_map_train = load_pseudolabels(data, experiment, embedding_algorithm, cluster_algorithm,
                                                    selected_k, "train")
        np.save(os.path.join(data, "topographical_map", "cluster_labels",
                             "_".join((data, "train", labels_filename_suffix))),
                topographical_map_train)

        if validation_set:
            # Load map of the validation set
            topographical_map_valid = load_pseudolabels(data, experiment, embedding_algorithm, cluster_algorithm,
                                                       selected_k, "valid")

            np.save(os.path.join(data, "topographical_map", "cluster_labels",
                                                            "_".join((data, "valid", labels_filename_suffix))),
                    topographical_map_valid)

        # Load map of the test set
        topographical_map_test = load_pseudolabels(data, experiment, embedding_algorithm, cluster_algorithm,
                                                   selected_k, "test")
        np.save(os.path.join(data, "topographical_map", "cluster_labels",
                                                        "_".join((data, "test", labels_filename_suffix))),
                topographical_map_test)

    print(f"Topographical Map produced and saved in {data}/topographical_map")
    print("-------------------------------------------------------------------------")
    print("Dataset name: ", data)
    print("Embedding: ", embedding_algorithm)
    print("Clustering algorithm: ", cluster_algorithm)
    print("Number of clusters: ", selected_k)
    return







