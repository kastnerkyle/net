# -*- coding: utf 8 -*-
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import gzip
import tarfile
import tempfile
import os
import numpy as np
from scipy import linalg
import glob
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
import warnings
import sys
# Sandbox?
from theano.tensor.shared_randomstreams import RandomStreams


def relu(x):
    return x * (x > 1e-6)


def clip_relu(x, clip_lim=20):
    return x * (T.lt(x, 1e-6) and T.gt(x, clip_lim))


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


def shared_zeros(shape):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(value=np.zeros(*shape).astype(theano.config.floatX),
                         borrow=True)


def shared_rand(shape, rng):
    """ Builds a theano shared variable filled with random values """
    return theano.shared(value=(rng.rand(*shape) - 0.5).astype(
        theano.config.floatX), borrow=True)


def shared_ortho(shape, rng, name=None):
    """ Builds a theano shared variable filled with random values """
    g = rng.randn(*shape)
    o_g = linalg.svd(g)[0]
    return theano.shared(value=o_g.astype(theano.config.floatX), borrow=True)


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

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
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
    train_x = np.asarray(train_data)
    train_y = np.asarray(train_labels)
    test_x = np.asarray(test_data)
    test_y = np.asarray(test_labels)
    valid_x = train_x[-10000:]
    valid_y = train_y[-10000:]
    train_x = train_x[:-10000]
    train_y = train_y[:-10000]

    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    valid_x = valid_x.astype('float32')
    valid_y = valid_y.astype('int32')
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_scribe():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'scribe.pkl'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/scribe.pkl'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    with open(data_file, 'rb') as pkl_file:
        data = cPickle.load(pkl_file)

    n_classes = data['nChars']
    data_x, data_y = [], []
    for x, y in zip(data['x'], data['y']):
        # Need to make alternate characters blanks (index as nClasses)
        y1 = [n_classes]
        for char in y:
            y1 += [char, n_classes]
        data_y.append(np.asarray(y1, dtype=np.int32))
        data_x.append(np.asarray(x, dtype=theano.config.floatX).T)

    train_x = data_x[:750]
    train_y = data_y[:750]
    valid_x = data_x[750:900]
    valid_y = data_y[750:900]
    test_x = data_x[900:]
    test_y = data_y[900:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
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


class TrainingMixin(object):
    def get_sgd_trainer(self, X_sym, y_sym, params, cost, learning_rate,
                        momentum):
        """ Returns a simple sgd trainer."""
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        for n, (param, gparam) in enumerate(zip(params, gparams)):
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * gparam
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step

        train_fn = theano.function(inputs=[X_sym, y_sym],
                                   outputs=cost,
                                   updates=updates)
        return train_fn

    def get_clip_sgd_trainer(self, X_sym, y_sym, params, cost, learning_rate,
                             momentum):
        """ Returns a simple sgd trainer."""
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        # Gradient clipping
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
        scaling_den = T.maximum(1., grad_norm)
        scaling_num = 1.
        for n, (param, gparam) in enumerate(zip(params, gparams)):
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * gparam * (
                scaling_num / scaling_den)
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step

        train_fn = theano.function(inputs=[X_sym, y_sym],
                                   outputs=cost,
                                   updates=updates)
        return train_fn


def build_linear_layer(input_size, output_size, input_variable, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    output_variable = T.dot(input_variable, W) + b
    params = [W, b]
    return output_variable, params


def build_tanh_layer(input_size, output_size, input_variable, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    output_variable = T.tanh(T.dot(input_variable, W) + b)
    params = [W, b]
    return output_variable, params


def build_relu_layer(input_size, output_size, input_variable, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    output_variable = relu(T.dot(input_variable, W) + b)
    params = [W, b]
    return output_variable, params


def build_sigmoid_layer(input_size, output_size, input_variable, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=4 * W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    output_variable = T.nnet.sigmoid(T.dot(input_variable, W) + b)
    params = [W, b]
    return output_variable, params


def softmax_cost(y_hat_sym, y_sym):
    return -T.mean(T.log(y_hat_sym)[T.arange(y_sym.shape[0]), y_sym])


class FeedforwardClassifier(BaseMinet, TrainingMixin):
    def __init__(self, hidden_layer_sizes=[500], batch_size=100, max_iter=1E3,
                 learning_rate=0.01, momentum=0., learning_alg="sgd",
                 activation="tanh", model_save_name="saved_model",
                 save_frequency=100, random_seed=None):

        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.max_iter = int(max_iter)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.model_save_name = model_save_name

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.learning_alg = learning_alg
        if activation == "relu":
            self.feedforward_function = build_relu_layer
        elif activation == "tanh":
            self.feedforward_function = build_tanh_layer
        elif activation == "sigmoid":
            self.feedforward_function = build_sigmoid_layer
        else:
            raise ValueError("Value %s not understood for activation"
                             % activation)

    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        input_variable = X_sym
        params = []
        for i, (input_size, output_size) in enumerate(zip(layer_sizes[:-1],
                                                          layer_sizes[1:-1])):
            output_variable, layer_params = self.feedforward_function(
                input_size, output_size, input_variable, self.random_state)
            params.extend(layer_params)
            input_variable = output_variable

        output_variable, layer_params = build_linear_layer(
            layer_sizes[-2], layer_sizes[-1], input_variable, self.random_state)
        params.extend(layer_params)
        y_hat_sym = T.nnet.softmax(output_variable)
        cost = softmax_cost(y_hat_sym, y_sym)

        self.params = params

        if self.learning_alg == "sgd":
            self.fit_function = self.get_sgd_trainer(X_sym, y_sym, params, cost,
                                                     self.learning_rate,
                                                     self.momentum)
        else:
            raise ValueError("Algorithm %s is not "
                             "a valid argument for learning_alg!"
                             % self.learning_alg)
        self.loss_function = theano.function(
            inputs=[X_sym, y_sym], outputs=cost)

        self.predict_function = theano.function(
            inputs=[X_sym],
            outputs=[y_hat_sym],)

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
        self.training_loss_ = []
        self.validation_loss_ = []

        if not hasattr(self, 'fit_function'):
            self._setup_functions(X_sym, y_sym,
                                  self.layer_sizes_)

        batch_indices = list(range(0, X.shape[0], self.batch_size))
        if X.shape[0] != batch_indices[-1]:
            batch_indices.append(X.shape[0])

        best_valid_loss = np.inf
        for itr in range(self.max_iter):
            print("Starting pass %d through the dataset" % itr)
            batch_bounds = list(zip(batch_indices[:-1], batch_indices[1:]))
            # Random minibatches
            self.random_state.shuffle(batch_bounds)
            for start, end in batch_bounds:
                self.partial_fit(X[start:end], y[start:end])
            current_train_loss = self.loss_function(X, y)
            self.training_loss_.append(current_train_loss)

            if (itr % self.save_frequency) == 0 or (itr == self.max_iter):
                f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            if valid_X is not None:
                current_valid_loss = self.loss_function(valid_X, valid_y)
                self.validation_loss_.append(current_valid_loss)
                print("Validation loss %f" % current_valid_loss)
                # if we got the best validation score until now, save
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    f = open(self.model_save_name + "_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
        return self

    def predict(self, X):
        return np.argmax(self.predict_function(X), axis=1)


def build_recurrent_tanh_layer(input_size, hidden_size, output_size,
                               input_variable, random_state):
    wih = shared_rand((input_size, hidden_size), random_state)
    whh = shared_ortho((hidden_size, hidden_size), random_state)
    who = shared_rand((hidden_size, output_size), random_state)
    bh = shared_zeros((hidden_size,))
    h0 = shared_zeros((hidden_size,))
    bo = shared_zeros((output_size,))
    params = [wih, bh, whh, h0, who, bo]

    def step(x_t, h_tm1):
        h_t = T.tanh(T.dot(h_tm1, whh) + T.dot(x_t, wih) + bh)
        return h_t

    hidden, _ = theano.scan(
        step,
        sequences=[input_variable],
        outputs_info=[h0]
    )
    return hidden, params


def build_recurrent_relu_layer(input_size, hidden_size, output_size,
                               input_variable, random_state):
    wih = shared_rand((input_size, hidden_size), random_state)
    whh = shared_ortho((hidden_size, hidden_size), random_state)
    who = shared_rand((hidden_size, output_size), random_state)
    bh = shared_zeros((hidden_size,))
    h0 = shared_zeros((hidden_size,))
    bo = shared_zeros((output_size,))
    params = [wih, bh, whh, h0, who, bo]

    def step(x_t, h_tm1):
        h_t = clip_relu(T.dot(h_tm1, whh) + T.dot(x_t, wih) + bh)
        return h_t

    hidden, _ = theano.scan(
        step,
        sequences=[input_variable],
        outputs_info=[h0]
    )
    return hidden, params


def build_recurrent_lstm_layer(input_size, hidden_size, output_size,
                               input_variable, random_state):
    h0 = shared_zeros((hidden_size,))
    c0 = shared_zeros((hidden_size,))
    who = shared_rand((hidden_size, output_size), random_state)
    bo = shared_zeros((output_size,))

    # Input gate weights
    wxig = shared_rand((hidden_size, input_size), random_state)
    whig = shared_ortho((hidden_size, hidden_size), random_state)
    wcig = shared_ortho((hidden_size, hidden_size), random_state)

    # Forget gate weights
    wxfg = shared_rand((hidden_size, input_size), random_state)
    whfg = shared_ortho((hidden_size, hidden_size), random_state)
    wcfg = shared_ortho((hidden_size, hidden_size), random_state)

    # Output gate weights
    wxog = shared_rand((hidden_size, input_size), random_state)
    whog = shared_ortho((hidden_size, hidden_size), random_state)
    wcog = shared_ortho((hidden_size, hidden_size), random_state)

    # Cell weights
    wxc = shared_rand((hidden_size, input_size), random_state)
    whc = shared_ortho((hidden_size, hidden_size), random_state)

    # Input gate bias
    big = shared_zeros((hidden_size,))

    # Forget gate bias
    bfg = shared_zeros((hidden_size,))

    # Output gate bias
    bog = shared_zeros((hidden_size,))

    # Cell bias
    bc = shared_zeros((hidden_size,))

    params = [wxig, whig, wcig,
              wxfg, whfg, wcfg,
              wxog, whog, wcog,
              wxc, whc, big, bfg, bog, bc,
              h0, c0, who, bo]

    def step(x_t, h_tm1, c_tm1):
        i_t = T.nnet.sigmoid(T.dot(wxig, x_t) + T.dot(whig, h_tm1) +
                             T.dot(wcig, c_tm1) + big)
        f_t = T.nnet.sigmoid(T.dot(wxfg, x_t) + T.dot(whfg, h_tm1) +
                             T.dot(wcfg, c_tm1) + bfg)
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(wxc, x_t) +
                                         T.dot(whc, h_tm1) + bc)
        o_t = T.nnet.sigmoid(T.dot(wxog, x_t) + T.dot(whog, h_tm1) +
                             T.dot(wcog, c_t) + bog)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    [hidden, cell], _ = theano.scan(step,
                                    sequences=[input_variable],
                                    outputs_info=[h0, c0])
    return hidden, cell, params


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


def ctc_cost(y_hat_sym, y_sym):
    forward_probs = path_probs(y_hat_sym, y_sym)
    backward_probs = path_probs(y_hat_sym[::-1], y_sym[::-1])[::-1, ::-1]
    probs = forward_probs * backward_probs / y_hat_sym[:, y_sym]
    total_probs = T.sum(probs)
    return -T.log(total_probs)


def slab_print(slab):
    """
    Prints a 'slab' of alignment using ascii.

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


class RecurrentCTC(BaseMinet, TrainingMixin):
    """
    CTC cost based on code by Shawn Tan.
    """
    def __init__(self, hidden_layer_sizes=[100], max_iter=1E2,
                 learning_rate=0.01, momentum=0., learning_alg="sgd",
                 recurrent_activation="tanh", feedforward_activation="tanh",
                 save_frequency=10, model_save_name="saved_model",
                 random_seed=None):
        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.learning_rate = learning_rate
        self.learning_alg = learning_alg
        self.momentum = momentum
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = int(max_iter)
        self.save_frequency = save_frequency
        self.model_save_name = model_save_name
        self.recurrent_activation = recurrent_activation
        if recurrent_activation == "tanh":
            self.recurrent_function = build_recurrent_tanh_layer
        elif recurrent_activation == "relu":
            self.recurrent_function = build_recurrent_relu_layer
        elif recurrent_activation == "lstm":
            self.recurrent_function = build_recurrent_lstm_layer
        else:
            raise ValueError("Value %s not understood for recurrent_activation"
                             % recurrent_activation)
        if feedforward_activation == "tanh":
            self.feedforward_activation = T.tanh
        elif feedforward_activation == "relu":
            self.feedforward_activation = relu
        else:
            raise ValueError("Value %s not understood" % feedforward_activation,
                             "for feedforward_activation")

    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        input_variable = X_sym
        if len(layer_sizes) % 2 == 0:
            # If there aren't the right number of layer sizes, add a layer of
            # the same size as the output
            warnings.warn("Length of layer_sizes needs to be odd!\n"
                          "Adding output layer of size %i" % layer_sizes[-1])
            layer_sizes.append(layer_sizes[-1])

        # Iterate in chunks of 3 creating recurrent layers
        for i in range(0, len(layer_sizes) - 2, 2):
            current_layers = layer_sizes[i:i+3]
            input_size, hidden_size, output_size = current_layers
            if self.recurrent_activation != "lstm":
                hidden, params = self.recurrent_function(
                    input_size, hidden_size, output_size, input_variable,
                    self.random_state)
            else:
                hidden, cell, params = self.recurrent_function(
                    input_size, hidden_size, output_size, input_variable,
                    self.random_state)

            Wo = params[-2]
            bo = params[-1]
            # Need last activation to be linear for CTC cost
            if i == (len(layer_sizes) - 3):
                input_variable = T.dot(hidden, Wo) + bo
            else:
                input_variable = self.feedforward_activation(
                    T.dot(hidden, Wo) + bo)

        y_hat_sym = T.nnet.softmax(input_variable)
        cost = ctc_cost(y_hat_sym, y_sym)

        self.params = params

        if self.learning_alg == "sgd":
            self.fit_function = self.get_clip_sgd_trainer(X_sym, y_sym, params,
                                                          cost,
                                                          self.learning_rate,
                                                          self.momentum)
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

        best_valid_loss = np.inf
        for itr in range(self.max_iter):
            print("Starting pass %d through the dataset" % itr)
            total_train_loss = 0
            for n in range(len(X)):
                train_loss = self.fit_function(X[n], y[n])
                total_train_loss += train_loss
            self.training_loss_.append(total_train_loss / len(X))

            if (itr % self.save_frequency) == 0 or (itr == self.max_iter):
                f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            if valid_X is not None:
                total_valid_loss = 0
                for n in range(len(valid_X)):
                    valid_loss = self.loss_function(valid_X[n], valid_y[n])
                    total_valid_loss += valid_loss
                current_valid_loss = total_valid_loss / len(valid_X)
                print("Validation loss %f" % current_valid_loss)
                self.validation_loss_.append(current_valid_loss)
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    f = open(self.model_save_name + "_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()

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
