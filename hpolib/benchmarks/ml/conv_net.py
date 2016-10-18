import time
import numpy as np
import lasagne
import theano
import theano.tensor as T

import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark


class ConvolutionalNeuralNetwork(AbstractBenchmark):
    """
        This benchmark implements the 5D hyperparameter optimization of a 3 layer convolutional neural network.
        Each layer consists of a convolution  with batch normalization and ReLU activation functions
        followed by max pooling. We use a filter size of 5x5 for all convolutions (similar to the CudaConvNet).
        The weights are optimized with Adam.
        The tunable hyperparameters are the learning rate (on a log scale), the batch size and
        the number of units in each layer (on a log2 scale).
    """
    def __init__(self, path=None, max_num_epochs=40):

        self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets = self.get_data(path)
        self.max_num_epochs = max_num_epochs
        self.num_classes = len(np.unique(self.train_targets))
        super(ConvolutionalNeuralNetwork, self).__init__()

    def get_data(self, path):
        pass

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, steps=1, **kwargs):

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        lc_curve, cost_curve, train_loss, valid_loss = self.train_net(self.train, self.train_targets,
                                                                      self.valid, self.valid_targets,
                                                                      init_learning_rate=np.power(10., x[0]),
                                                                      batch_size=int(x[1]),
                                                                      n_units_1=int(np.power(2, x[2])),
                                                                      n_units_2=int(np.power(2, x[3])),
                                                                      n_units_3=int(np.power(2, x[4])),
                                                                      num_epochs=num_epochs)

        y = lc_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y,
                "cost": c,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_curve": lc_curve,
                "learning_curve_cost": cost_curve}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, steps=1, **kwargs):

        num_epochs = int(1 + (self.max_num_epochs - 1) * steps)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        lc_curve, cost_curve, train_loss, valid_loss = self.train_net(train, train_targets,
                                                                      self.test, self.test_targets,
                                                                      init_learning_rate=np.power(10., x[0]),
                                                                      batch_size=int(x[1]),
                                                                      n_units_1=int(np.power(2, x[2])),
                                                                      n_units_2=int(np.power(2, x[3])),
                                                                      n_units_3=int(np.power(2, x[4])),
                                                                      num_epochs=num_epochs)
        y = lc_curve[-1]
        c = cost_curve[-1]
        return {'function_value': y,
                "cost": c,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_curve": lc_curve,
                "learning_curve_cost": cost_curve}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=np.random.randint(1, 100000))
        cs.generate_all_continuous_from_bounds(ConvolutionalNeuralNetwork.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Convolutional Neural Network',
                'bounds': [[-6, 0],  # init_learning_rate
                           [32, 512],  # batch_size
                           [4, 10],  # n_units_1
                           [4, 10],  # n_units_2
                           [4, 10]  # n_units_3
                           ]}

    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

    def train_net(self, train, train_targets,
                  valid, valid_targets,
                  init_learning_rate=3*1e-5,
                  batch_size=256,
                  n_units_1=128,
                  n_units_2=128,
                  n_units_3=128,
                  num_epochs=140):

        start_time = time.time()

        input_var = T.ftensor4('inputs')
        target_var = T.ivector('targets')

        # Build net
        network = lasagne.layers.InputLayer(shape=(None, train.shape[1],  train.shape[2],  train.shape[3]),
                                            input_var=input_var)

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,
                                                                       num_filters=n_units_1,
                                                                       filter_size=(5, 5),
                                                                       pad="same",
                                                                       stride=1,
                                                                       W=lasagne.init.HeNormal(),
                                                                       b=lasagne.init.Constant(val=0.0),
                                                                       nonlinearity=lasagne.nonlinearities.rectify))

        network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,
                                                                       num_filters=n_units_2,
                                                                       filter_size=(5, 5),
                                                                       pad="same",
                                                                       stride=1,
                                                                       W=lasagne.init.HeNormal(),
                                                                       b=lasagne.init.Constant(val=0.0),
                                                                       nonlinearity=lasagne.nonlinearities.rectify))

        network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network,
                                                                       num_filters=n_units_3,
                                                                       filter_size=(5, 5),
                                                                       pad="same",
                                                                       stride=1,
                                                                       W=lasagne.init.HeNormal(),
                                                                       b=lasagne.init.Constant(val=0.0),
                                                                       nonlinearity=lasagne.nonlinearities.rectify))

        network = lasagne.layers.MaxPool2DLayer(network, pool_size=3, stride=2)

        network = lasagne.layers.DenseLayer(network, num_units=self.num_classes,
                                            nonlinearity=lasagne.nonlinearities.softmax)

        # Define Theano functions
        params = lasagne.layers.get_all_params(network, trainable=True)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           target_var)
        loss = loss.mean()

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()

        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        learning_rate = theano.shared(np.float32(init_learning_rate))

        updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        print("Starting training...")

        learning_curve = np.zeros([num_epochs])
        cost = np.zeros([num_epochs])
        train_loss = np.zeros([num_epochs])
        valid_loss = np.zeros([num_epochs])

        for e in range(num_epochs):

            epoch_start_time = time.time()
            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(train, train_targets, batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(valid, valid_targets, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(e + 1, num_epochs, time.time() - epoch_start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

            learning_curve[e] = 1 - val_acc / val_batches
            cost[e] = time.time() - start_time
            train_loss[e] = train_err / train_batches
            valid_loss[e] = val_err / val_batches

        return learning_curve, cost, train_loss, valid_loss
