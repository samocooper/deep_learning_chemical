import numpy
from scipy.sparse import csc_matrix
import h5py
import sys
import theano
import theano.tensor as T

from sda.utils import load_data, print_array
from sda.SdA import SdA


def main():

    print('Loading data from file...')
    f = h5py.File('feature_matrix.h5', 'r')
    data = f['data'][:]
    col = f['col'][:]
    row = f['row'][:]
    shape = f['shape'][:]


    matrix = csc_matrix((numpy.array(data), (numpy.array(row), numpy.array(col))), shape=(shape[0], shape[1]),
                        dtype=numpy.uint8)

    print(matrix.shape)

    # exit(1)

    batch_size = 10
    n_samples, n_vars =  matrix.shape
    n_train_batches = n_samples / batch_size

    numpy_rng = numpy.random.RandomState(23432)

    # build model
    print('Building model...')

    sda = SdA(numpy_rng=numpy_rng, n_ins=n_vars,
              hidden_layers_sizes=[int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])])

    print('configuring...')
    pretraining_fns = sda.pretraining_functions(train_set_x=matrix.todense(), batch_size=batch_size)

    print('training...')
    pretraining_epochs = 15
    pretrain_lr = 0.001
    corruption_levels = [0.1, 0.2, 0.3] + [0.4] * sda.n_layers
    for i in xrange(sda.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))

    y = sda.get_lowest_hidden_values(matrix)
    get_y = theano.function([], y)
    y_val = get_y()
    print_array(y_val, index=len(sys.argv)-1)

if __name__ == '__main__':
    main()

