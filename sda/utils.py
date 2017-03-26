""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.
"""

import numpy
import pandas as pd

import theano
import theano.tensor as T


def print_array(X, sep='\t', index=None):
    """ Prints array to standard out in csv style format.
    """
    if not index:
        for row in X:
            line = sep.join(map(str, row))
            print(line)

        return

    for i, row in zip(index, X):
        line = str(i) + sep + sep.join(map(str, row))
        print(line)


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset.
    '''

    #############
    # LOAD DATA #
    #############

    df = pd.read_table(dataset)
    data = df.ix[:, 1:].as_matrix()
    index = list(df.ix[:, 0])

    def shared_dataset(data, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data = data
        shared_data = theano.shared(numpy.asarray(data,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_data

    data_set = shared_dataset(data)
    return data_set, index
