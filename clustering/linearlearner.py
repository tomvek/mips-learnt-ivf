"""
    *project*:
     mips-learnt-ivf

    *authors*:
     Thomas Vecchiato, Claudio Lucchese, Franco Maria Nardini, Sebastian Bruch

    *name file*:
     linearlearner.py

    *version file*:
     1.0

    *description*:
     Linear-learner algorithm (our method).
"""

import numpy as np
import pandas as pd
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam


def nn_linear(k, input_shape, n_units):
    """
    Neural network model definition, with one input layer (dimension of the embedding),
    one hidden layer (optional) and one output layer (number of clusters).

    :param k: The number of clusters.
    :param input_shape: The dimension of the embedding.
    :param n_units: The number of units in the hidden layer;
     if it is set to 0, the hidden layer is removed, having only the input and output layer.
    :return: The neural network model.
    """
    # w/out hidden layer
    if n_units == 0:
        model = models.Sequential()
        model.add(layers.Dense(k, activation='softmax', use_bias=False, input_shape=input_shape))
        return model

    # w/ hidden layer
    else:
        model = models.Sequential()
        model.add(layers.Dense(n_units, activation=None, use_bias=False, input_shape=input_shape))
        model.add(layers.Dense(k, activation='softmax', use_bias=False))
        return model


def run_nn(n_clusters, input_shape, x_train, y_train, x_val, y_val, n_epochs=50, n_units=0):
    """
    Trains the neural network.

    :param n_clusters: The number of clusters.
    :param input_shape: The dimension of the embedding.
    :param x_train, y_train: Train set.
    :param x_val, y_val: Validation set.
    :param n_epochs: The number of epochs to train our neural network.
    :param n_units: The number of units in the hidden layer;
     if it is set to 0, the hidden layer is removed, having only the input and output layer.
    :return: The neural network model and the history of the neural network model.
    """
    # model definition
    model_nn = nn_linear(n_clusters, input_shape, n_units)

    # compiling the model
    model_nn.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])

    # training the model
    history = model_nn.fit(x_train, y_train,
                           epochs=n_epochs,
                           batch_size=512,
                           validation_data=(x_val, y_val))

    return model_nn, history


def run_linear_learner(x_train, y_train, x_val, y_val, train_queries, n_clusters, n_epochs, n_units):
    """
    Main function to run the linear-learner algorithm (our method).

    :param x_train, y_train: Train set.
    :param x_val, y_val: Validation set.
    :param train_queries: The queries used to train our model.
    :param n_clusters: The number of clusters.
    :param n_epochs: The number of epochs to train our neural network.
    :param n_units: The number of units in the hidden layer;
     if it is set to 0, the hidden layer is removed, having only the input and output layer.
    :return: The new computed centroid, for each cluster.
    """
    # neural network model
    print('Running linear learner, with number of units: ', n_units)
    _, history = run_nn(n_clusters,
                        (train_queries.shape[1],),
                        x_train, y_train,
                        x_val, y_val,
                        n_epochs=n_epochs,
                        n_units=n_units)

    df_history = pd.DataFrame(history.history)
    with open('history_nn.json', mode='w') as f:
        df_history.to_json(f)

    # best number of epochs
    best_n_epochs = list(range(n_epochs))[np.argmin(history.history['val_loss'])]
    print('Best number of epochs: ', best_n_epochs)

    model_nn, _ = run_nn(n_clusters,
                         (train_queries.shape[1],),
                         x_train, y_train,
                         x_val, y_val,
                         n_epochs=best_n_epochs,
                         n_units=n_units)

    # return the new centroids
    if n_units == 0:
        return model_nn.get_weights()[0].T
    else:
        return np.matmul(model_nn.get_weights()[0], model_nn.get_weights()[1]).T
