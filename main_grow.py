import os
import sys
import timeit

import numpy
import h5py
import theano
import theano.tensor as T
from sklearn.metrics import roc_auc_score


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, name, file):

        weights = []
        biases = []

        for model in file:
            name_temp = model
            if name_temp.split()[0] == 'WH1':
                weights.append(file[model][...])
            if name_temp.split()[0] == 'bH1':
                biases.append(file[model][...])

        # Initialise weights add previous weights

        self.input = input

        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

        for i in range(len(weights)):
            W_values = numpy.concatenate((W_values, weights[i]), 1)
            b_values = numpy.concatenate((b_values, biases[i]))

        self.W = theano.shared(value=W_values, name='WH1 ' + name, borrow=True)
        self.b = theano.shared(value=b_values, name='bH1 ' + name, borrow=True)

        # Mask so only new section is updated

        W_mask = numpy.zeros(W_values.shape, dtype=theano.config.floatX)
        b_mask = numpy.zeros(b_values.shape, dtype=theano.config.floatX)

        W_mask[:, :n_out] = 1
        b_mask[:n_out] = 1

        self.Wmask = theano.shared(value=W_mask, name='MaskW1' + name, borrow=True)
        self.bmask = theano.shared(value=b_mask, name='Maskb1' + name, borrow=True)

        # Momentum for weights
        self.Wm = theano.shared(numpy.zeros(W_values.shape, dtype=theano.config.floatX), name='Wm', borrow=True)
        self.bm = theano.shared(numpy.zeros(b_values.shape, dtype=theano.config.floatX), name='bm', borrow=True)

        lin_output = T.dot(input, self.W) + self.b
        self.output = (T.nnet.relu(lin_output))

        # parameters of the model
        self.params = [self.W, self.b]
        self.mask = [self.Wmask, self.bmask]
        self.m = [self.Wm, self.bm]

        self.shape = W_values.shape

    def get_shape(self):
        return self.shape

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, name, file):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)


        weights = []
        biases = []
        flag = False
        for model in file:
            name_temp = model
            if name_temp.split()[0] == 'WC1':
                weights.append(file[model][...])
            if name_temp.split()[0] == 'bC1':
                biases.append(file[model][...])
                flag = True

        # Initialise weights add previous weights

        self.input = input

        W_values = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
        print(W_values.shape)

        for i in range(len(weights)):
            W_values = numpy.concatenate((W_values, weights[i]), 0)
        self.W = theano.shared(value=W_values, name='WC1 '+name, borrow=True)

        print(W_values.shape)

        # initialize the biases b as a vector of n_out 0s
        if flag:
            b_values = biases[0]
        else:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='bC1 '+name, borrow=True)

        W_mask = numpy.zeros(W_values.shape, dtype=theano.config.floatX)
        b_mask = numpy.zeros(b_values.shape, dtype=theano.config.floatX)

        W_mask[:n_in, :] = 1
        b_mask[:] = 1

        self.Wmask = theano.shared(value=W_mask, name='MaskW1' + name, borrow=True)
        self.bmask = theano.shared(value=b_mask, name='Maskb1' + name, borrow=True)

        self.Wm = theano.shared(numpy.zeros(W_values.shape, dtype=theano.config.floatX), name='Wm', borrow=True)
        self.bm = theano.shared(numpy.zeros(b_values.shape, dtype=theano.config.floatX), name='bm', borrow=True)

        # Final sigmoid function

        self.y_pred = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]
        self.mask = [self.Wmask, self.bmask]
        self.m = [self.Wm, self.bm]

        # keep track of model input
        self.input = input

    def cross_entropy(self, y):
        return T.mean(T.nnet.binary_crossentropy(self.y_pred, y).mean())

    def errors(self, y):
        return T.neq(self.y_pred > 0.5, y)


def load_data(hdf5_data, hdf5_test):

    # Training data
    data = h5py.File(hdf5_data, 'a')

    data_x = data['x_activity'][...]
    data_y = data['y_activity'][...]

    data_y[data_y < 0] = 0
    data_x[data_x < 0] = 0

    train_set_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=True)

    # Testing data

    data_test = h5py.File(hdf5_test, 'a')

    data_x_test = data_test['x_activity_test'][...]
    data_y_test = data_test['y_activity_test'][...]

    data_y_test[data_y_test < 0] = 0
    data_x_test[data_x_test < 0] = 0

    valid_set_x = theano.shared(numpy.asarray(data_x_test, dtype=theano.config.floatX), borrow=True)
    valid_set_y = theano.shared(numpy.asarray(data_y_test, dtype=theano.config.floatX), borrow=True)

    return [train_set_x, train_set_y, valid_set_x, valid_set_y, data_x.shape[1]]


def sgd_optimization_mnist(hdf5_name='hdf5_file.hdf5', name='_', learning_rate=5,
                           n_epochs=5, batch_size=200, hdf5_data='_',
                           hdf5_test='_'):

    model_file = h5py.File(hdf5_name, 'a')
    train_set_x, train_set_y, valid_set_x, valid_set_y, n_in = load_data(hdf5_data, hdf5_test)

    # compute number of minibatches for training, validation and testing

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    ###############
    # BUILD MODEL #
    ###############

    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.matrix('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28

    rng = numpy.random.RandomState(1234)

    # Layers of the model
    new_layers = 50

    hiddenLayer1 = HiddenLayer(
        rng=rng,
        input=x,
        n_in=n_in,
        n_out=new_layers,
        name=name,
        file=model_file
    )
    shape = hiddenLayer1.get_shape()
    classifier = LogisticRegression(input=hiddenLayer1.output,
                                    n_in=new_layers,
                                    n_out=7,
                                    name=name,
                                    file=model_file)

    params = hiddenLayer1.params + classifier.params
    masks = hiddenLayer1.mask + classifier.mask
    momentum = hiddenLayer1.m + classifier.m

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.cross_entropy(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch

    test_model = theano.function(
        inputs=[index],
        outputs=[cost, classifier.y_pred],
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in params]

    updates = []
    for i in range(len(gparams)):
        if len(params[i].get_value(borrow=True).shape) == 2:
            param_i = (params[i], (params[i] - learning_rate * gparams[i] - momentum[i]) * masks[i])
            moment_i = (momentum[i], (momentum[i] * 0.8 + learning_rate * gparams[i]) * masks[i])
            updates.append(param_i)
            updates.append(moment_i)

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ################################
    # TEST ROC AUC BEFORE TRAINING #
    ################################

    data_test = h5py.File('data_matrix_test.hdf5', 'a')
    data_y_test = data_test['y_activity_test'][...]
    data_y_test[data_y_test < 0] = 0

    pred_y = numpy.zeros(data_y_test.shape)
    for i in range(n_valid_batches):
        cost, pred_y_temp = test_model(i)
        pred_y[i * batch_size: (i + 1) * batch_size, :] = pred_y_temp

    print('ROC AUC IS', roc_auc_score(data_y_test.flatten(), pred_y.flatten()))

    ###############
    # TRAIN MODEL #
    ###############

    print(params[0].name)
    #print(params[0].get_value(borrow=True))

    print('... training the model')

    validation_frequency = 10
    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        train_cost = 0

        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            train_cost += minibatch_avg_cost

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

        if epoch % validation_frequency == 0:

            # compute zero-one loss on validation set
            # validation_losses = [validate_model(i) for i in range(n_valid_batches)]
            # this_validation_loss = numpy.sum(validation_losses)

            print(epoch, train_cost)
        else:
            print(epoch, train_cost)

    ################
    # TEST ROC AUC #
    ################

    pred_y = numpy.zeros(data_y_test.shape)
    for i in range(n_valid_batches):
        cost, pred_y_temp = test_model(i)
        pred_y[i * batch_size: (i + 1) * batch_size, :] = pred_y_temp

    print('ROC AUC IS', roc_auc_score(data_y_test.flatten(), pred_y.flatten()))

    for param in params:
        name_temp = param.name

        for model in model_file:
            if model == name_temp:
                del model_file[model]

        model_file.create_dataset(name=name_temp, data=param.get_value(borrow=True))


if __name__ == '__main__':
    sgd_optimization_mnist(name='sam', hdf5_name='current_model_file.hdf5', hdf5_data='data_matrix.hdf5', hdf5_test='data_matrix_test.hdf5')