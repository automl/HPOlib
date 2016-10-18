import time
import numpy as np
import lasagne
import theano
import theano.tensor as T

from hpolib.abstract_benchmark import AbstractBenchmark

import ConfigSpace as CS


class LogisticRegression(AbstractBenchmark):
    """
        Logistic regression benchmark which resembles the benchmark used in the MTBO / Freeze Thaw paper.
        The hyperparameters are the learning rate (on log scale), L2 regularization, batch size and the
        dropout ratio on the inputs.
        The weights are optimized with stochastic gradient descent and we do NOT perform early stopping on
        the validation data set.
    """

    def __init__(self, path=None):
        self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets = self.get_data(path)
        self.num_classes = len(np.unique(self.train_targets))
        self.num_epochs = 100
        super(LogisticRegression, self).__init__()

    def get_data(self, path):
        pass

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, **kwargs):

        start_time = time.time()

        # Shuffle training data
        shuffle = np.random.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        # Split of dataset subset
        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        learning_rate = np.float32(10 ** x[0])
        l2_reg = np.float32(x[1])
        batch_size = np.int32(x[2])
        dropout_rate = np.float32(x[3])

        lc_curve, cost_curve = self.run(train=train,
                                        train_targets=train_targets,
                                        valid=self.valid,
                                        valid_targets=self.valid_targets,
                                        learning_rate=learning_rate,
                                        l2_reg=l2_reg,
                                        batch_size=batch_size,
                                        dropout_rate=dropout_rate,
                                        num_epochs=self.num_epochs)
        y = lc_curve[-1]
        c = time.time() - start_time

        return {'function_value': y, "cost": c, "learning_curve": lc_curve, "cost_curve": cost_curve}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):

        start_time = time.time()

        learning_rate = np.float32(10 ** x[0])
        l2_reg = np.float32(x[1])
        batch_size = np.int32(x[2])
        dropout_rate = np.float32(x[3])

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        lc_curve, cost_curve = self.run(train=train,
                                        train_targets=train_targets,
                                        valid=self.test,
                                        valid_targets=self.test_targets,
                                        learning_rate=learning_rate,
                                        l2_reg=l2_reg,
                                        batch_size=batch_size,
                                        dropout_rate=dropout_rate,
                                        num_epochs=self.num_epochs)
        y = lc_curve[-1]
        c = time.time() - start_time

        return {'function_value': y, "cost": c, "learning_curve": lc_curve, "cost_curve": cost_curve}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace(seed=np.random.randint(1, 100000))
        cs.generate_all_continuous_from_bounds(LogisticRegression.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Logistic Regression',
                'bounds': [[-6, 0],  # learning rate
                           [0, 1],  # l2 regularization
                           [20, 2000],  # batch size
                           [0, .75]  # dropout rate
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

    def run(self, train, train_targets,
            valid, valid_targets,
            learning_rate=1, l2_reg=0.0,
            batch_size=200, dropout_rate=0.1, num_epochs=100):

        start_time = time.time()

        input_var = T.dmatrix('inputs')
        target_var = T.ivector('targets')

        # Build net
        lr = lasagne.layers.InputLayer(shape=(None, train.shape[1]),
                                       input_var=input_var)

        lr = lasagne.layers.DropoutLayer(lr, p=dropout_rate)

        lr = lasagne.layers.DenseLayer(lr,
                                       num_units=self.num_classes,
                                       W=lasagne.init.HeNormal(),
                                       b=lasagne.init.Constant(val=0.0),
                                       nonlinearity=lasagne.nonlinearities.softmax)

        # Define Theano functions
        params = lasagne.layers.get_all_params(lr, trainable=True)
        prediction = lasagne.layers.get_output(lr)
        loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                           target_var)
        # Add L2 regularization for the weights
        l2_penalty = l2_reg * lasagne.regularization.regularize_network_params(lr, lasagne.regularization.l2)

        loss += l2_penalty
        loss = loss.mean()

        test_prediction = lasagne.layers.get_output(lr, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()

        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        learning_rate = theano.shared(learning_rate)

        updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)

        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        print("Starting training...")

        learning_curve = np.zeros([num_epochs])
        cost = np.zeros([num_epochs])

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

        return learning_curve, cost
