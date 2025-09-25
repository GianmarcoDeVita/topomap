import os

def generate_reports(data):

    # Save cumulative results with report of cluster accuracy
    file_out = open(os.path.join(data, "reports", '_'.join(("topomap", "report", "clustering", '.'.join((data, "csv"))))), "w")
    file_out.write("Embedding,Clustering,K,Silhouette,DB_Scores,CH_Scores,ClusterAccuracy\n")
    file_out.flush()

    # Save cumulative results with report of pairwise weighted accuracy
    file_sel = open(os.path.join(data, "reports", '_'.join(("topomap", "report", "configurations_pairwise", '.'.join((data, "csv"))))), "w")
    file_sel.write("Embedding,Clustering,K,PairWiseMinAccuracy,PairWiseMeanAccuracy\n")
    file_sel.flush()

    # Save pairwise accuracies
    file_pw_acc = open(os.path.join(data, "reports", '_'.join(("topomap", "report", "pairwise", "accuracies",
                                                         '.'.join((data, "csv"))))), "w")
    file_pw_acc.write("Embedding,Clustering,K,Experiment,Cluster1,Cluster2,PairWiseAccuracy\n")
    file_pw_acc.flush()

    # Save number of components by experiment
    file_ncomp = open(os.path.join(data, "reports", '_'.join(("topomap", "report", "ncomp", '.'.join((data, "csv"))))), "w")
    file_ncomp.write("Embedding,Components\n")
    file_ncomp.flush()

    return file_out, file_sel, file_pw_acc, file_ncomp
