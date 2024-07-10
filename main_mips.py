"""
    **A Learning-to-Rank Formulation of Clustering-Based Approximate Nearest Neighbor Search**

    *project*:
     mips-learnt-ivf

    *authors*:
     Thomas Vecchiato, Claudio Lucchese, Franco Maria Nardini, Sebastian Bruch

    *name file*:
     main_mips.py

    *version file*:
     1.0

    *description*:
     A critical piece of the modern information retrieval puzzle is approximate nearest neighbor search.
     Its objective is to return a set of k data points that are closest to a query point, with its accuracy measured by
     the proportion of exact nearest neighbors captured in the returned set. One popular approach to this question
     is clustering: The indexing algorithm partitions data points into non-overlapping subsets and represents each
     partition by a point such as its centroid. The query processing algorithm first identifies the nearest
     clusters — a process known as routing — then performs a nearest neighbor search over those clusters only.
     In this work, we make a simple observation: The routing function solves a ranking problem.
     Its quality can therefore be assessed with a ranking metric, making the function amenable to learning-to-rank.
     Interestingly, ground-truth is often freely available: Given a query distribution in a top-k configuration,
     the ground-truth is the set of clusters that contain the exact top-k vectors. We develop this insight and apply
     it to Maximum Inner Product Search (MIPS). As we demonstrate empirically on various datasets,
     learning a simple linear function consistently improves the accuracy of clustering-based MIPS.

    *run commands*:

     1. **hdf5 format**:
     python main_mips.py --name_dataset ... --name_embedding ... --format_file hdf5 --dataset ... --algorithm ...
     --nclusters ... --top_k ... --test_split_percent ... --split_seed ... --ells ... --learner_nunits ...
     --learner_nepochs ... --compute_clusters ...

     2. **npy format**:
     python main_mips.py --name_dataset ... --name_embedding ... --format_file npy --dataset_docs ...
     --dataset_queries ... --dataset_neighbors ... --algorithm ... --nclusters ... --top_k ... --test_split_percent ...
     --split_seed ... --ells ... --learner_nunits ... --learner_nepochs ... --compute_clusters ...

"""

import numpy as np
import h5py
from absl import app, flags
import time
from tabulate import tabulate
from clustering import kmeans, k_random, linearlearner, auxiliary
from tqdm import tqdm


# names of the algorithms
AlgorithmRandom = 'random'
AlgorithmKMeans = 'kmeans'
AlgorithmSphericalKmeans = 'kmeans-spherical'
AlgorithmLinearLearner = 'linear-learner'

# name of the dataset and embedding
flags.DEFINE_string('name_dataset', None, 'Name of the dataset.')
flags.DEFINE_string('name_embedding', None, 'Name of the embedding.')

# decide the file format to import
flags.DEFINE_string('format_file', None, 'hdf5 - for the hdf5 file; npy - for the npy files.')

# dataset for hdf5
flags.DEFINE_string('dataset', None, 'Path to the dataset in hdf5 format.')

flags.DEFINE_string('documents_key', 'documents', 'Dataset key for document vectors.')
flags.DEFINE_string('train_queries_key', 'train_queries', 'Dataset key for train queries.')
flags.DEFINE_string('valid_queries_key', 'valid_queries', 'Dataset key for validation queries.')
flags.DEFINE_string('test_queries_key', 'test_queries', 'Dataset key for test queries.')
flags.DEFINE_string('train_neighbors_key', 'train_neighbors', 'Dataset key for train neighbors.')
flags.DEFINE_string('valid_neighbors_key', 'valid_neighbors', 'Dataset key for validation neighbors.')
flags.DEFINE_string('test_neighbors_key', 'test_neighbors', 'Dataset key for test neighbors.')

# docs, queries and neighbors for npy
flags.DEFINE_string('dataset_docs', None, 'Path to the dataset-docs in npy format.')
flags.DEFINE_string('dataset_queries', None, 'Path to the dataset-queries in npy format.')
flags.DEFINE_string('dataset_neighbors', None, 'Path to the dataset-neighbors in npy format.')

# setting environment
flags.DEFINE_float('test_split_percent', 20, 'Percentage of data points in the test set.')
flags.DEFINE_integer('split_seed', 42, 'Seed used when forming train-test splits.')

# linear-learner
flags.DEFINE_integer('learner_nunits', 0, 'Number of hidden units used by the linear-learner, with 0 we drop'
                                             'the hidden layer.')
flags.DEFINE_integer('learner_nepochs', 100, 'Number of epochs used by the linear-learner.')

# algorithm method
flags.DEFINE_enum('algorithm', AlgorithmKMeans,
                  [AlgorithmRandom,
                   AlgorithmKMeans,
                   AlgorithmSphericalKmeans,
                   AlgorithmLinearLearner],
                  'Indexing algorithm.')

flags.DEFINE_integer('nclusters', 1000, 'When `algorithm` is KMeans-based: Number of clusters.')

# multi-probing, set the probes
flags.DEFINE_list('ells', [1],
                  'Minimum number of documents to examine.')

# top-k docs
flags.DEFINE_integer('top_k', 1, 'Top-k documents to retrieve per query.')

# flag to skip the clustering algorithm if already computed
flags.DEFINE_integer('compute_clusters', 0, '0 - perform clustering algorithm; '
                                            '1 - take the results already computed.')

FLAGS = flags.FLAGS


def get_final_results(name_method, centroids, x_test, y_test, top_k, clusters_top_k_test=None, gpu_flag=True):
    """
    Computes the final results, where we have the accuracy of given centroids.

    :param name_method: Name of the method that generated the centroids under consideration.
    :param centroids: The centroids.
    :param x_test, y_test: The test set.
    :param top_k: The number of top documents.
    :param clusters_top_k_test: The clusters where the top documents for each query are located.
    :param gpu_flag: Flag that indicates whether to run the code using the GPU (True) or not (False).
    """

    # compute the score for each query and centroid
    print(name_method, end=' ')
    print('- run prediction with centroids...', end=' ')
    pred = auxiliary.scores_queries_centroids(centroids, x_test, gpu_flag=gpu_flag)
    print('end, shape: ', pred.shape)

    # save scores computed
    print('Saving results: score for each query and centroid.')
    np.save('./ells_stat_sig/' + name_method + '_' + FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' +
            FLAGS.algorithm + '_ells_stat_sig.npy', pred)

    # computation of the final scores
    results_ells = []
    if top_k > 1:
        for threshold in tqdm(FLAGS.ells):
            k = int(threshold)
            one_pred = auxiliary.computation_top_k_clusters(k, FLAGS.nclusters, pred)
            res = auxiliary.evaluate_ell_top_k(one_pred, clusters_top_k_test, top_k)
            results_ells.append(res)
            print('k = {0}: {1}'.format(k, res))
    else:
        for threshold in tqdm(FLAGS.ells):
            k = int(threshold)
            one_pred = auxiliary.computation_top_k_clusters(k, FLAGS.nclusters, pred)
            res = auxiliary.evaluate_ell_top_one(one_pred, y_test)
            results_ells.append(res)
            print('k = {0}: {1}'.format(k, res))

    # print the final results
    table = ([['n_k', 'acc']] + [[FLAGS.ells[i_c], results_ells[i_c]] for i_c in range(len(FLAGS.ells))])
    print(tabulate(table, headers='firstrow', tablefmt='psql'))

    # save the results
    file_result = open('./results/' + name_method + '_' + FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' +
                       FLAGS.algorithm + str(top_k) + '_results.txt', 'w')
    file_result.write(tabulate(table, headers='firstrow', tablefmt='psql'))
    file_result.close()
    print('Results saved.')


def main(_):
    """
    Main function of our algorithm, where all methods to obtain the final results are invoked.
    """
    start = time.time()

    documents = None
    queries = None
    neighbors = None

    # hdf5 format file
    if FLAGS.format_file == 'hdf5':
        dataset = h5py.File(FLAGS.dataset, 'r')

        documents = np.array(dataset[FLAGS.documents_key])

        train_queries = None
        train_ground_truth = None
        valid_queries = None
        valid_ground_truth = None
        test_queries = None
        test_ground_truth = None

        # Prepare the dataset
        if FLAGS.train_queries_key in dataset:
            train_queries = dataset[FLAGS.train_queries_key]
            train_ground_truth = dataset[FLAGS.train_neighbors_key]
            train_ground_truth = train_ground_truth[:, :FLAGS.top_k]
        if FLAGS.valid_queries_key in dataset:
            valid_queries = dataset[FLAGS.valid_queries_key]
            valid_ground_truth = dataset[FLAGS.valid_neighbors_key]
            valid_ground_truth = valid_ground_truth[:, :FLAGS.top_k]
        if FLAGS.test_queries_key in dataset:
            test_queries = dataset[FLAGS.test_queries_key]
            test_ground_truth = dataset[FLAGS.test_neighbors_key]
            test_ground_truth = test_ground_truth[:, :FLAGS.top_k]

        queries = []
        neighbors = []
        if FLAGS.train_queries_key in dataset:
            queries.append(train_queries)
            neighbors.append(train_ground_truth)
        if FLAGS.valid_queries_key in dataset:
            queries.append(valid_queries)
            neighbors.append(valid_ground_truth)
        if FLAGS.test_queries_key in dataset:
            queries.append(test_queries)
            neighbors.append(test_ground_truth)

        neighbors = np.concatenate(neighbors, axis=0) if len(queries) > 1 else neighbors[0]
        queries = np.concatenate(queries, axis=0) if len(queries) > 1 else queries[0]

        assert len(queries) == len(neighbors)

    # npy format file
    elif FLAGS.format_file == 'npy':
        documents = np.load(FLAGS.dataset_docs)
        queries = np.load(FLAGS.dataset_queries)
        neighbors = np.load(FLAGS.dataset_neighbors)

        assert len(queries) == len(neighbors)

    # run the clustering algorithm or import the clusters already computed
    print('Running the clustering algorithm or importing the clusters already computed.')

    centroids = None
    label_clustering = None

    # compute centroids and labels
    if FLAGS.compute_clusters == 1:

        # (standard or spherical) kmeans algorithm
        if FLAGS.algorithm in [AlgorithmKMeans, AlgorithmSphericalKmeans]:
            spherical = FLAGS.algorithm == AlgorithmSphericalKmeans
            centroids, label_clustering = kmeans.k_means(doc_vectors=documents,
                                                         n_clusters=FLAGS.nclusters,
                                                         flag_spherical=spherical)
            print(f'Obtained centroids with shape: {centroids.shape}')

        # shallow kmeans algorithm
        elif FLAGS.algorithm == AlgorithmRandom:
            centroids, label_clustering = k_random.random_clustering(doc_vectors=documents,
                                                                     n_clusters=FLAGS.nclusters)
            print(f'Obtained centroids with shape: {centroids.shape}')

        # save centroids and label_clustering
        centroids_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_centroids.npy'
        label_clustering_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_label_clustering.npy'

        print('Saving clusters got.')
        np.save(centroids_file, centroids)
        np.save(label_clustering_file, label_clustering)

    # load centroids and labels
    else:
        centroids_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_centroids.npy'
        label_clustering_file = FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_label_clustering.npy'

        centroids = np.load(centroids_file)
        label_clustering = np.load(label_clustering_file)

    # data preparation
    print('Data preparation.')
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None
    y_test = None
    clusters_top_k_test = None

    if FLAGS.format_file == 'hdf5':
        partitioning = auxiliary.train_test_val(queries, neighbors, size_split=FLAGS.test_split_percent/100)

        x_train = partitioning[0]
        y_train = auxiliary.query_true_label(FLAGS.nclusters, label_clustering, partitioning[1][:, 0])
        x_val = partitioning[2]
        y_val = auxiliary.query_true_label(FLAGS.nclusters, label_clustering, partitioning[3][:, 0])
        x_test = partitioning[4]
        y_test = auxiliary.query_true_label(FLAGS.nclusters, label_clustering,  partitioning[5][:, 0])

        if FLAGS.top_k > 1:
            clusters_top_k = []
            for best_test_neigh in partitioning[5]:
                clusters_top_k.append(label_clustering[best_test_neigh])
            clusters_top_k_test = np.array(clusters_top_k)

    elif FLAGS.format_file == 'npy':
        label_data = auxiliary.query_true_label(FLAGS.nclusters, label_clustering, neighbors)
        partitioning = auxiliary.train_test_val(queries, label_data, size_split=FLAGS.test_split_percent/100)

        x_train = partitioning[0]
        y_train = partitioning[1]
        x_val = partitioning[2]
        y_val = partitioning[3]
        x_test = partitioning[4]
        y_test = partitioning[5]

    np.save(FLAGS.name_dataset + '_' + FLAGS.name_embedding + '_' + FLAGS.algorithm + '_y_test.npy', y_test)

    # training linear-learner
    print('Linear Learner.')
    new_centroids = linearlearner.run_linear_learner(x_train=x_train, y_train=y_train,
                                                     x_val=x_val, y_val=y_val,
                                                     train_queries=queries,
                                                     n_clusters=FLAGS.nclusters,
                                                     n_epochs=FLAGS.learner_nepochs,
                                                     n_units=FLAGS.learner_nunits)

    print(f'Obtained centroids with shape: {new_centroids.shape}')

    # results: baseline
    get_final_results('baseline', centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test, gpu_flag=True)

    # results: linear-learner
    get_final_results('linearlearner', new_centroids, x_test, y_test, FLAGS.top_k, clusters_top_k_test, gpu_flag=True)

    end = time.time()
    print(f'Done in {end - start} seconds.')


if __name__ == '__main__':
    app.run(main)
