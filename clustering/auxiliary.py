"""
    *project*:
     mips-learnt-ivf

    *authors*:
     Thomas Vecchiato, Claudio Lucchese, Franco Maria Nardini, Sebastian Bruch

    *name file*:
     auxiliary.py

    *version file*:
     1.0

    *description*:
     Auxiliary functions for our project, necessary to obtain the final results.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch


def scores_queries_centroids(centroids, x_test, gpu_flag=True):
    """
    Compute for each query the distribution of the centroids according the dot-product calculation (<.,.>),
    that can be performed w/ or w/out the GPU.

    :param centroids: The centroids.
    :param x_test: The embedding queries.
    :param gpu_flag: Flag that indicates whether to run the code using the GPU (True) or not (False).
    :return: The dot-product between each query and each centroid.
    """

    # gpu computation
    if gpu_flag:
        dim_iter = x_test.shape[0]

        torch.cuda.set_device(0)
        centroids = torch.from_numpy(centroids).cuda(0)
        x_test = torch.from_numpy(x_test).cuda(0)

        results = []
        for i in tqdm(range(dim_iter), leave=False, colour='cyan'):
            results.append(torch.mv(centroids, x_test[i]).to('cpu').numpy())
        return np.array(results)
    # cpu computation
    else:
        def score_computation(query, means=centroids):
            return np.array([np.dot(query, center) for center in means])

        return np.apply_along_axis(score_computation, axis=1, arr=x_test)


def create_vector_distribution(k, pos):
    """
    Create a vector with values 0 and 1, where the 1s are set according the indices given.

    :param k: The number of clusters.
    :param pos: The indices where to set the value to 1.
    :return: Vector with length equal to the number of clusters and with values 0 and 1.
    """

    vect = np.zeros(k)
    vect[pos] = 1.
    return vect


def query_true_label(n_cluster, label_clustering, neighbors):
    """
    Create the ground truth for each query, where the cell with value 1 indicates in which cluster the best document
    is located.

    :param n_cluster: The number of clusters.
    :param label_clustering: Vector where each position indicates the related document and the value inside the cell
     is the cluster where the document is located.
    :param neighbors: Vector where each position indicates the related query and the value inside the cell
     is the best document.
    :return: The ground truth.
    """

    return np.array([create_vector_distribution(n_cluster, label_clustering[i_doc].item())
                     for i_doc in neighbors], dtype=np.int8)


def train_test_val(query_vectors, label_data, size_split=0.2):
    """
    Data split in train, validation and test.

    :param query_vectors: The embedding queries.
    :param label_data: The labels of the queries.
    :param size_split: Floating number to indicate how split the data.
    :return: Train, validation and test set.
    """

    x_train, x_test, y_train, y_test = train_test_split(query_vectors, label_data,
                                                        test_size=size_split, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=size_split, random_state=42)
    return [x_train, y_train, x_val, y_val, x_test, y_test]


def computation_top_k_clusters(top_k, n_cluster, scores):
    """
    The computation of the top k clusters for each query, where the top-k clusters for each query is returned.

    :param top_k: The number of the top clusters that are analyzed.
    :param n_cluster: The number of clusters.
    :param scores: The dot-product score between each query and cluster.
    :return: The top-k clusters for each query.
    """

    # auxiliary function to compute computation_top_k_clusters
    def centroids_query(scores_vec, top_k=top_k):
        max_top_k_ind = np.argpartition(scores_vec, -top_k)[-top_k:]
        res_vec = np.zeros(n_cluster)
        res_vec[max_top_k_ind] = 1.
        return res_vec

    return np.apply_along_axis(centroids_query, axis=1, arr=scores)


def evaluate_ell_top_one(x_test, y_test):
    """
    Compute the accuracy regarding multi-probing top-1.

    :param x_test: Our result.
    :param y_test: Ground truth.
    :return: The accuracy.
    """

    correct = 0.

    for i in range(x_test.shape[0]):
        if np.sum(np.where(x_test[i] == 1)[0] == np.where(y_test[i] == 1)[0]) > 0:
            correct += 1.

    return correct / float(x_test.shape[0])


def evaluate_ell_top_k(x_test, y_test_list, top_k):
    """
    Compute the accuracy regarding multi-probing top-k.

    :param x_test: Our result.
    :param y_test_list: Ground truth.
    :param top_k: The number of top-k documents under consideration for a given query.
    :return: The accuracy.
    """

    correct = 0.

    for i in range(x_test.shape[0]):
        response_clusters = np.where(x_test[i] == 1)[0]
        correct += np.sum([np.count_nonzero(y_test_list[i] == el) for el in response_clusters]) / float(top_k)

    return correct / float(x_test.shape[0])
