try:
    import cPickle
except ImportError:
    import pickle as cPickle
import gzip
import tarfile
import tempfile
import os
import numpy as np
import time
import glob
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
import sys
# Sandbox?
from theano.tensor.shared_randomstreams import RandomStreams


def dropout(random_state, X, keep_prob=0.5):
    if keep_prob > 0. and keep_prob < 1.:
        seed = random_state.randint(2 ** 30)
        srng = RandomStreams(seed)
        mask = srng.binomial(n=1, p=keep_prob, size=X.shape,
                             dtype=theano.config.floatX)
        return X * mask
    return X


def fast_dropout(random_state, X):
    seed = random_state.randint(2 ** 30)
    srng = RandomStreams(seed)
    mask = srng.normal(size=X.shape, avg=1., dtype=theano.config.floatX)
    return X * mask


def shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(value=np.zeros(shape, dtype=theano.config.floatX),
                         name=name, borrow=True)


def shared_rand(shape, rng, name=None):
    """ Builds a theano shared variable filled with random values """
    return theano.shared(value=(rng.rand(*shape) - 0.5).astype(
        theano.config.floatX))


def load_mnist():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'mnist.pkl.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    test_set_x = test_set_x.astype('float32')
    test_set_y = test_set_y.astype('int32')
    valid_set_x, valid_set_y = valid_set
    valid_set_x = valid_set_x.astype('float32')
    valid_set_y = valid_set_y.astype('int32')
    train_set_x, train_set_y = train_set
    train_set_x = train_set_x.astype('float32')
    train_set_y = train_set_y.astype('int32')

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_cifar10():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'cifar-10-python.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    tar = tarfile.open(data_file)
    os.chdir(data_path)
    tar.extractall()
    tar.close()

    data_path = os.path.join(data_path, "cifar-10-batches-py")
    batch_files = glob.glob(os.path.join(data_path, "*batch*"))
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for f in batch_files:
        batch_file = open(f, 'rb')
        d = cPickle.load(batch_file)
        batch_file.close()
        fname = f.split(os.path.sep)[-1]
        if "data" in fname:
            data = d['data']
            labels = d['labels']
            train_data.append(data)
            train_labels.append(labels)
        elif "test" in fname:
            data = d['data']
            labels = d['labels']
            test_data.append(data)
            test_labels.append(labels)

    # Split into 40000 train 10000 valid 10000 test
    train_set_x = np.asarray(train_data)
    train_set_y = np.asarray(train_labels)
    test_set_x = np.asarray(test_data)
    test_set_y = np.asarray(test_labels)
    valid_set_x = train_set_x[-10000:]
    valid_set_y = train_set_y[-10000:]
    train_set_x = train_set_x[:-10000]
    train_set_y = train_set_y[:-10000]

    test_set_x = test_set_x.astype('float32')
    test_set_y = test_set_y.astype('int32')
    valid_set_x = valid_set_x.astype('float32')
    valid_set_y = valid_set_y.astype('int32')
    train_set_x = train_set_x.astype('float32')
    train_set_y = train_set_y.astype('int32')

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


class BaseMinet(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class BaseFeedforward(object):
    def __init__(self, hidden_layer_sizes, batch_size, max_iter,
                 random_seed, save_frequency, model_save_name):
        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.max_iter = max_iter
        self.save_frequency = save_frequency
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.model_save_name = model_save_name

    def partial_fit(self, X, y):
        return self.fit_function(X, y.astype('int32'))

    def fit(self, X, y, valid_X=None, valid_y=None):
        input_size = X.shape[1]
        output_size = len(np.unique(y))
        X_sym = T.matrix('x')
        y_sym = T.ivector('y')
        self.layers_ = []
        self.layer_sizes_ = [input_size]
        self.layer_sizes_.extend(self.hidden_layer_sizes)
        self.layer_sizes_.append(output_size)
        self.dropout_layers_ = []
        self.training_scores_ = []
        self.validation_scores_ = []
        self.training_loss_ = []
        self.validation_loss_ = []

        if not hasattr(self, 'fit_function'):
            self._setup_functions(X_sym, y_sym,
                                  self.layer_sizes_)

        batch_indices = list(range(0, X.shape[0], self.batch_size))
        if X.shape[0] != batch_indices[-1]:
            batch_indices.append(X.shape[0])

        start_time = time.clock()
        itr = 0
        best_validation_score = np.inf
        while (itr < self.max_iter):
            print("Starting pass %d through the dataset" % itr)
            itr += 1
            batch_bounds = list(zip(batch_indices[:-1], batch_indices[1:]))
            # Random minibatches
            self.random_state.shuffle(batch_bounds)
            for start, end in batch_bounds:
                self.partial_fit(X[start:end], y[start:end])
            current_training_score = (self.predict(X) != y).mean()
            self.training_scores_.append(current_training_score)
            current_training_loss = self.loss_function(X, y)
            self.training_loss_.append(current_training_loss)
            # Serialize each save_frequency iteration
            if (itr % self.save_frequency) == 0 or (itr == self.max_iter):
                f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()
            if valid_X is not None:
                current_validation_score = (
                    self.predict(valid_X) != valid_y).mean()
                self.validation_scores_.append(current_validation_score)
                current_training_loss = self.loss_function(valid_X, valid_y)
                self.validation_loss_.append(current_training_loss)
                print("Validation score %f" % current_validation_score)
                # if we got the best validation score until now, save
                if current_validation_score < best_validation_score:
                    best_validation_score = current_validation_score
                    f = open(self.model_save_name + "_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
        end_time = time.clock()
        print("Total training time ran for %.2fm" %
              ((end_time - start_time) / 60.))
        return self

    def predict(self, X):
        return self.predict_function(X)


class TrainingMixin(object):
    def get_sgd_trainer(self, X_sym, y_sym, params, cost, learning_rate):
        """ Returns a simple sgd trainer."""
        gparams = T.grad(cost, params)
        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            updates[param] = param - learning_rate * gparam

        train_fn = theano.function(inputs=[X_sym, y_sym],
                                   outputs=cost,
                                   updates=updates)
        return train_fn

    def get_adagrad_trainer(self, X_sym, y_sym, params, cost, learning_rate,
                            adagrad_param):
        gparams = T.grad(cost, params)
        self.accumulated_gradients_ = []
        accumulated_gradients_ = self.accumulated_gradients_

        for layer in self.layers_:
            accumulated_gradients_.extend([shared_zeros(p.shape.eval(),
                                           'accumulated_gradient')
                                           for p in layer.params])
        updates = OrderedDict()
        for agrad, param, gparam in zip(accumulated_gradients_,
                                        params, gparams):
            ag = agrad + gparam * gparam
            # TODO: Norm clipping
            updates[param] = param - (learning_rate / T.sqrt(
                ag + adagrad_param)) * gparam
            updates[agrad] = ag

        train_fn = theano.function(inputs=[X_sym, y_sym],
                                   outputs=cost,
                                   updates=updates)
        return train_fn


# TODO: replace with softmax_cost?
class Softmax(BaseMinet):
    def __init__(self, input_variable, n_in=None, n_out=None, weights=None,
                 biases=None):
        if weights is None:
            assert n_in is not None
            assert n_out is not None
            W = theano.shared(value=np.zeros((n_in, n_out),
                                             dtype=theano.config.floatX),
                              name='W', borrow=True)
            b = theano.shared(value=np.zeros((n_out,),
                                             dtype=theano.config.floatX),
                              name='b', borrow=True)
        else:
            W = weights
            b = biases

        self.W = W
        self.b = b
        self.p_y_given_x = T.nnet.softmax(T.dot(input_variable, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred')
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


# TODO: Replace with build_hidden_layer?
class HiddenLayer(BaseMinet):
    def __init__(self, input_variable, rng, n_in=None, n_out=None, weights=None,
                 biases=None, activation=T.tanh):
        self.input_variable = input_variable
        if not weights:
            assert n_in is not None
            assert n_out is not None
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            W = weights
            b = biases

        self.W = W
        self.b = b

        linear_output = T.dot(self.input_variable, self.W) + self.b
        self.output = (linear_output if activation is None
                       else activation(linear_output))
        self.params = [self.W, self.b]


class MLP(BaseMinet, BaseFeedforward, TrainingMixin):
    def __init__(self, hidden_layer_sizes=[500], batch_size=100, max_iter=1E3,
                 dropout=True, learning_rate=0.01, l1_reg=0., l2_reg=1E-4,
                 learning_alg="sgd", adagrad_param=1E-6, adadelta_param=0.9,
                 activation="tanh", model_save_name="saved_model",
                 save_frequency=100, random_seed=None):

        super(MLP, self).__init__(hidden_layer_sizes=hidden_layer_sizes,
                                  batch_size=batch_size, max_iter=max_iter,
                                  random_seed=random_seed,
                                  save_frequency=save_frequency,
                                  model_save_name=model_save_name)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.learning_alg = learning_alg
        self.adagrad_param = adagrad_param
        self.adadelta_param = adadelta_param
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        if activation == "relu":
            def relu(x):
                return x * (x > 1e-6)
            self.activation = relu
        elif activation == "tanh":
            self.activation = T.tanh
        elif activation == "sigmoid":
            self.activation = T.nnet.sigmoid
        else:
            raise ValueError("Value %s not understood for activation"
                             % activation)

    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        input_variable = X_sym
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:-1])):
            keep_prob = 0.8 if i == 0 else 0.5
            if not self.dropout:
                keep_prob = 1.0
            self.layers_.append(HiddenLayer(
                rng=self.random_state,
                input_variable=keep_prob * input_variable,
                n_in=n_in, n_out=n_out,
                activation=self.activation))

            dropout_input_variable = dropout(self.random_state, input_variable,
                                             keep_prob=keep_prob)
            W, b = self.layers_[-1].params
            self.dropout_layers_.append(HiddenLayer(
                rng=self.random_state,
                input_variable=dropout_input_variable,
                weights=W, biases=b,
                activation=self.activation))

            input_variable = self.layers_[-1].output

        keep_prob = 0.5
        if self.dropout:
            keep_prob = 1.0
        self.layers_.append(Softmax(input_variable=keep_prob * input_variable,
                                    n_in=layer_sizes[-2],
                                    n_out=layer_sizes[-1]))

        dropout_input_variable = dropout(self.random_state, input_variable,
                                         keep_prob=keep_prob)
        W, b = self.layers_[-1].params
        self.dropout_layers_.append(Softmax(input_variable=dropout_input_variable,
                                    weights=W, biases=b,
                                    n_out=layer_sizes[-1]))

        self.l1 = 0
        for hl in self.layers_:
            self.l1 += abs(hl.W).sum()

        self.l2_sqr = 0.
        for hl in self.layers_:
            self.l2_sqr += (hl.W ** 2).sum()

        self.negative_log_likelihood = self.dropout_layers_[-1].negative_log_likelihood

        self.params = self.layers_[0].params
        for hl in self.layers_[1:]:
            self.params += hl.params
        self.cost = self.negative_log_likelihood(y_sym)
        self.cost += self.l1_reg * self.l1
        self.cost += self.l2_reg * self.l2_sqr

        self.errors = self.layers_[-1].errors
        self.loss_function = theano.function(
            inputs=[X_sym, y_sym], outputs=self.negative_log_likelihood(y_sym))

        self.predict_function = theano.function(
            inputs=[X_sym], outputs=self.layers_[-1].y_pred)

        if self.learning_alg == "sgd":
            self.fit_function = self.get_sgd_trainer(X_sym, y_sym, self.params,
                                                     self.cost,
                                                     self.learning_rate)
        elif self.learning_alg == "adagrad":
            self.fit_function = self.get_adagrad_trainer(X_sym, y_sym,
                                                         self.params,
                                                         self.cost,
                                                         self.learning_rate,
                                                         self.adagrad_param)
        else:
            raise ValueError("Algorithm %s is not "
                             "a valid argument for learning_alg!"
                             % self.learning_alg)


def build_recurrent_layer(inpt, wih, whh, bh, h0):
    def step(x_t, h_tm1):
        h_t = T.tanh(T.dot(h_tm1, whh) + T.dot(x_t, wih) + bh)
        return h_t

    hidden, _ = theano.scan(
        step,
        sequences=[inpt],
        outputs_info=[h0]
    )
    return hidden


def recurrence_relation(size):
    eye2 = T.eye(size + 2)
    return T.eye(size) + eye2[2:, 1:-1] + eye2[2:, :-2] * (T.arange(size) % 2)


def path_probs(predict, y_sym):
    pred_y = predict[:, y_sym]
    rr = recurrence_relation(y_sym.shape[0])

    def step(p_curr, p_prev):
        return p_curr * T.dot(p_prev, rr)

    probabilities, _ = theano.scan(
        step,
        sequences=[pred_y],
        outputs_info=[T.eye(y_sym.shape[0])[0]]
    )
    return probabilities


def ctc_cost(predict, y_sym):
    forward_probs = path_probs(predict, y_sym)
    backward_probs = path_probs(predict[::-1], y_sym[::-1])[::-1, ::-1]
    probs = forward_probs * backward_probs / predict[:, y_sym]
    total_probs = T.sum(probs)
    return -T.log(total_probs)


def slab_print(slab):
    """
    Prints a 'slab' of printed 'text' using ascii.
    slab: A matrix of floats from [0, 1]

    Originally from
    https://github.com/rakeshvar/rnn_ctc
    """
    for ir, r in enumerate(slab):
        sys.stdout.write('{:2d}|'.format(ir))
        for val in r:
            if val == 0:
                sys.stdout.write(' ')
            elif val < .25:
                sys.stdout.write('░')
            elif val < .5:
                sys.stdout.write('▒')
            elif val < .75:
                sys.stdout.write('▓')
            elif val < 1.0:
                sys.stdout.write('█')
            else:
                sys.stdout.write('█')
        print('|')


def rnn_check_array(X):
    if type(X) == np.ndarray and len(X.shape) == 2:
        return [X]
    elif type(X) == np.ndarray and len(X.shape) == 3:
        return X
    try:
        X[0].shape[1]
    except AttributeError:
        raise ValueError("X must be an iterable of 2D numpy arrays")
    return X


class RNN_CTC(BaseMinet, TrainingMixin):
    """
    CTC cost based on code by Shawn Tan.
    """
    def __init__(self, hidden_layer_sizes=[9], max_iter=100,
                 learning_rate=0.01, learning_alg="sgd", adagrad_param=1E-6,
                 random_seed=1999):
        self.learning_rate = learning_rate
        self.adagrad_param = adagrad_param
        self.learning_alg = learning_alg
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = np.random.RandomState(random_seed)
        self.max_iter = max_iter

    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        input_sz = layer_sizes[0]
        hidden_sz = layer_sizes[1]
        output_sz = layer_sizes[-1]
        wih = shared_rand((input_sz, hidden_sz), self.random_state)  # input to hidden
        whh = shared_rand((hidden_sz, hidden_sz), self.random_state)  # hidden to hidden
        # whh is a matrix means all hidden units are fully inter-connected
        who = shared_rand((hidden_sz, output_sz), self.random_state)  # hidden to output
        bh = shared_rand((hidden_sz,), self.random_state)
        h0 = shared_rand((hidden_sz,), self.random_state)
        bo = shared_rand((output_sz,), self.random_state)
        params = [wih, whh, who, bh, h0, bo]

        hidden = build_recurrent_layer(X_sym, wih, whh, bh, h0)

        y_hat_sym = T.nnet.softmax(T.dot(hidden, who) + bo)
        cost = ctc_cost(y_hat_sym, y_sym)

        if self.learning_alg == "sgd":
            self.fit_function = self.get_sgd_trainer(X_sym, y_sym, params,
                                                     cost,
                                                     self.learning_rate)
        else:
            raise ValueError("Value of %s not a valid learning_alg!"
                             % self.learning_alg)

        self.loss_function = theano.function(inputs=[X_sym, y_sym],
                                             outputs=cost)

        self.predict_function = theano.function(
            inputs=[X_sym],
            outputs=[y_hat_sym],)

    def fit(self, X, y, valid_X=None, valid_y=None):
        X = rnn_check_array(X)
        input_size = X[0].shape[1]
        output_size = np.max([len(np.unique(d)) for d in y])
        X_sym = T.matrix('x')
        y_sym = T.ivector('y')
        self.layers_ = []
        self.layer_sizes_ = [input_size]
        self.layer_sizes_.extend(self.hidden_layer_sizes)
        self.layer_sizes_.append(output_size)
        if not hasattr(self, 'fit_function'):
            self._setup_functions(X_sym, y_sym,
                                  self.layer_sizes_)
        self.training_loss_ = []
        if valid_X is not None:
            self.validation_loss_ = []
        for itr in range(self.max_iter):
            print("Starting pass %d through the dataset" % itr)
            average_train_loss = 0
            for n in range(len(X)):
                train_loss = self.fit_function(X[n], y[n])
                average_train_loss += train_loss
            self.training_loss_.append(average_train_loss / len(X))

            if valid_X is not None:
                average_valid_loss = 0
                for n in range(len(valid_X)):
                    valid_loss = self.loss_function(valid_X[n], valid_y[n])
                    average_valid_loss += valid_loss
                self.validation_loss_.append(average_valid_loss / len(valid_X))

    def predict(self, X):
        X = rnn_check_array(X)
        blank_pred = self.layer_sizes_[-1] - 1
        predictions = []
        for n in range(len(X)):
            pred = np.argmax(self.predict_function(X[n])[0], axis=1)
            out = []
            for n, p in enumerate(pred):
                if (p != blank_pred) and (n == 0 or p != pred[n-1]):
                    out.append(p)
            predictions.append(np.array(out))
        return predictions

    def predict_proba(self, X):
        X = rnn_check_array(X)
        predictions = []
        for n in range(len(X)):
            pred = self.predict_function(X[n])
            predictions.append(pred)
        return predictions

    def print_alignment(self, X):
        X = rnn_check_array(X)
        for n in range(len(X)):
            print("Alignment for sample %d" % n)
            pred = self.predict_function(X[n])[0]
            slab_print(X[n].T)
            slab_print(pred.T[:-1])
