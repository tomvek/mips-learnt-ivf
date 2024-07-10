"""
    *project*:
     mips-learnt-ivf

    *authors*:
     Thomas Vecchiato, Claudio Lucchese, Franco Maria Nardini, Sebastian Bruch

    *name file*:
     kmeans.py

    *version file*:
     1.0

    *description*:
     Standard and Spherical KMeans clustering algorithms.
"""

import numpy as np
import faiss


def k_means(doc_vectors, n_clusters, flag_spherical=True):
    """
    Standard and Spherical KMeans clustering.

    :param doc_vectors: All embedding documents.
    :param n_clusters: The number of clusters.
    :param flag_spherical: Flag that indicates whether to run Spherical (True) or Standard (False) Kmeans.
    :return: The centroid of each cluster and for each document it returns the corresponding cluster.
    """

    clustering = faiss.Kmeans(doc_vectors.shape[1],
                              n_clusters,
                              spherical=flag_spherical,
                              gpu=True)

    samples = np.random.choice(doc_vectors.shape[0], min(39 * n_clusters, doc_vectors.shape[0]), replace=False)
    clustering.train(doc_vectors[samples])

    _, label_clustering = clustering.index.search(doc_vectors, 1)

    return clustering.centroids, label_clustering
