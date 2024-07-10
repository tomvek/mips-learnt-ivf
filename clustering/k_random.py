"""
    *project*:
     mips-learnt-ivf

    *authors*:
     Thomas Vecchiato, Claudio Lucchese, Franco Maria Nardini, Sebastian Bruch

    *name file*:
     k_random.py

    *version file*:
     1.0

    *description*:
     Shallow KMeans clustering algorithm.
"""

import random
import numpy as np
import torch
from tqdm import tqdm


def random_clustering(doc_vectors, n_clusters):
    """
    Shallow KMeans clustering algorithm.

    :param doc_vectors: All embedding documents.
    :param n_clusters: The number of clusters.
    :return: The centroid of each cluster and for each document it returns the corresponding cluster.
    """

    torch.cuda.set_device(0)

    n_docs = doc_vectors.shape[0]

    main_members_cpu = doc_vectors[random.sample(range(0, n_docs), n_clusters)]
    main_members = torch.from_numpy(main_members_cpu).cuda(0)
    doc_vectors = torch.from_numpy(doc_vectors).cuda(0)

    results = []
    for i in tqdm(range(n_docs), leave=False, colour='cyan'):
        results.append(torch.argmax(torch.mv(main_members, doc_vectors[i])).item())

    return main_members_cpu, np.array(results)


# """
#     shallow kmeans clustering algorithm - cpu version
# """
# def random_clustering(doc_vectors, n_clusters):
#     n_docs = doc_vectors.shape[0]
#     main_members = doc_vectors[random.sample(range(0, n_docs), n_clusters)]
#
#     def compute_label_clustering_doc(doc, main_members=main_members):
#         return np.argmax(main_members.dot(doc))
#
#     return main_members, np.apply_along_axis(compute_label_clustering_doc, axis=1, arr=doc_vectors)
