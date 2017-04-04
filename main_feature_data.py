import numpy
import h5py
import theano
import theano.tensor as T
from sklearn.metrics import roc_auc_score

# Neural network hidden fully connected layer

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out):
        self.input = input

        # Initialise weights and biases

        bounds = numpy.sqrt(6. / (n_in + n_out))

        W_values = numpy.asarray(rng.uniform(low=-bounds, high=bounds, size=(n_in, n_out)), dtype=theano.config.floatX)
        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

        # Set weights and biases as theano shared variables for GPU acceleration

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # Momentum terms to improve learning rate and avoid local optima

        self.Wm = theano.shared(numpy.zeros(W_values.shape, dtype=theano.config.floatX), name='Wm', borrow=True)
        self.bm = theano.shared(numpy.zeros(b_values.shape, dtype=theano.config.floatX), name='bm', borrow=True)

        # Linear rectifier unit activation function for sparse data

        lin_output = T.dot(input, self.W) + self.b
        self.output = (T.nnet.relu(lin_output))

        # Combine Parameters and momentum terms of the model

        self.params = [self.W, self.b]
        self.m = [self.Wm, self.bm]

# Neural network classification and error determination final layer


class CrossEntropyLoss(object):

    def __init__(self, input, n_in, n_out):
        self.input = input

        # Initialize weights and bias terms

        W_values = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

        # Store as numpy shared variables

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # Initialise and store momentum terms

        self.Wm = theano.shared(numpy.zeros(W_values.shape, dtype=theano.config.floatX), name='Wm', borrow=True)
        self.bm = theano.shared(numpy.zeros(b_values.shape, dtype=theano.config.floatX), name='bm', borrow=True)

        # Sigmoid activation function for final layer classification

        self.y_pred = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # Combine parameters and momentum terms

        self.params = [self.W, self.b]
        self.m = [self.Wm, self.bm]

    def cross_entropy(self, y):

        # Return cross entropy loss value

        return T.mean(T.nnet.binary_crossentropy(self.y_pred, y).mean())

    def errors(self, y):

        # Return errors made at some threshold level

        return T.neq(self.y_pred > 0.5, y)

# Data class to manage training and test object


class DataObject(object):

    def __init__(self, x_file_name, y_file_name, batch_size, mini_batch_size):

        # Load in training data set

        data_x = h5py.File(x_file_name, 'r')
        self.data_x = data_x['x_features'][...].astype(int)

        data_y = h5py.File(y_file_name, 'r')
        self.data_y = data_y['y_activity'][...]

        self.dims = self.data_x.shape
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        # pad matrix so all data is used in training

        pad = batch_size - (self.dims[0] % batch_size)
        print('Row number: ', self.dims[0], 'Number of padded rows: ', pad)

        self.data_x = numpy.concatenate((self.data_x, numpy.zeros((pad, self.dims[1])).astype(int)), axis=0)
        self.data_y = numpy.concatenate((self.data_y, numpy.zeros((pad, self.data_y.shape[1]))), axis=0)
        self.dims = self.data_x.shape

        self.feature_number = int(numpy.max(self.data_x.flatten()))+1
        print(self.feature_number)
        # Create matrix containing y_values to be paired with x_values

        self.x_row_values = numpy.zeros((batch_size, self.dims[1]))
        for i in range(batch_size):
            self.x_row_values[i, :] = i
        self.x_row_values = self.x_row_values.astype(int).flatten()

        # Vectorise matrix

        data_x_batch = numpy.zeros((batch_size, self.feature_number))
        data_x_batch[self.x_row_values, self.data_x[:batch_size, :].flatten()] = 1

        self.shared_x = theano.shared(numpy.asarray(data_x_batch, dtype=theano.config.floatX), borrow=True)
        self.shared_y = theano.shared(numpy.asarray(self.data_y[:batch_size, :], dtype=theano.config.floatX), borrow=True)

        self.n_batches = self.dims[0] // batch_size
        self.n_mini_batches = batch_size // mini_batch_size

    def new_shared(self, index):

        pos1 = self.batch_size*index
        pos2 = self.batch_size*(index+1)

        data_x_batch = numpy.zeros((self.batch_size, self.feature_number))
        data_x_batch[self.x_row_values, self.data_x[pos1:pos2, :].flatten()] = 1

        self.shared_x.set_value(numpy.asarray(data_x_batch, dtype=theano.config.floatX), borrow=True)
        self.shared_y.set_value(numpy.asarray(self.data_y[pos1:pos2, :], dtype=theano.config.floatX), borrow=True)

    def roc_auc(self, test_model):

        pred_y = numpy.zeros(self.data_y.shape)

        for i in range(self.n_batches):

            self.new_shared(i)
            pos = self.batch_size * i
            for j in range(self.n_mini_batches):
                cost, pred_y_temp = test_model(j)
                pred_y[pos + j * self.mini_batch_size: pos + (j + 1) * self.mini_batch_size, :] = pred_y_temp

        print('ROC AUC IS', roc_auc_score(self.data_y.flatten(), pred_y.flatten()))

# Neural network setup class


class NeuralNetwork(object):

    def __init__(self, x, y, feature_number, learning_rate=0.5):

        rng = numpy.random.RandomState(1234)

        # Construct network

        self.hiddenLayer1 = HiddenLayer(rng=rng, input=x, n_in=feature_number, n_out=50)
        self.classifier = CrossEntropyLoss(input=self.hiddenLayer1.output, n_in=50, n_out=7)

        # combine network parameters and momentum terms

        self.params = self.hiddenLayer1.params + self.classifier.params
        self.momentum = self.hiddenLayer1.m + self.classifier.m

        # define the cost as the cross entropy
        self.cost = self.classifier.cross_entropy(y)

        # Compile gradients and updates for neural network
        gradients = [T.grad(self.cost, param) for param in self.params]
        self.updates = [(param_i, param_i - learning_rate * grad_i - moment_i) for param_i, grad_i, moment_i in
                   zip(self.params, gradients, self.momentum)] + \
                  [(moment_i, moment_i * 0.8 + learning_rate * grad_i) for moment_i, grad_i in
                   zip(self.momentum, gradients)]


def run_network(n_epochs=35, mini_batch_size=200):

    #############
    # LOAD DATA #
    #############

    # Load data determine number of inputs to neural network

    train_data = DataObject('features_train.hdf5', 'activities_train.hdf5', 10000, mini_batch_size)
    #test_data = DataObject('activities_test.hdf5', 1000, mini_batch_size)

    # compute number of batches for training, validation and testing

    n_in = train_data.dims[1]

    ###############
    # BUILD MODEL #
    ###############

    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input

    x = T.matrix('x')  # assay data
    y = T.matrix('y')  # predicted values

    # function to test the model and output mistakes at specific threshold and soft classification values

    nnet = NeuralNetwork(x, y, train_data.feature_number)

    '''
    test_model = theano.function(
        inputs=[index],
        outputs=[nnet.cost, nnet.classifier.y_pred],
        givens={
            x: test_data.shared_x[index * mini_batch_size: (index + 1) * mini_batch_size],
            y: test_data.shared_y[index * mini_batch_size: (index + 1) * mini_batch_size]
        }
    )
    '''
    # function to test the model and return current cost

    train_model = theano.function(
        inputs=[index],
        outputs=nnet.cost,
        updates=nnet.updates,
        givens={
            x: train_data.shared_x[index * mini_batch_size: (index + 1) * mini_batch_size],
            y: train_data.shared_y[index * mini_batch_size: (index + 1) * mini_batch_size]
        }
    )

    ################################
    # TEST ROC AUC BEFORE TRAINING #
    ################################

    #test_data.roc_auc(test_model)

    ###############
    # TRAIN MODEL #
    ###############

    print('... training the model number of batches is', train_data.n_batches)

    done_looping = False
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):

        epoch += 1
        cost = 0
        for i in range(train_data.n_batches):

            train_data.new_shared(i)

            for j in range(train_data.n_mini_batches):
                cost += train_model(j)

            print(cost)
        #test_data.roc_auc(test_model)

    ################
    # TEST ROC AUC #
    ################

    #test_data.roc_auc(test_model)

    #f2 = h5py.File('y_pred01.hdf5', 'a')
    #f2.create_dataset('y_pred', data=pred_y)

if __name__ == '__main__':
    run_network()