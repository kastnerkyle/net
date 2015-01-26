# -*- coding: utf 8 -*-
from __future__ import division
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
from scipy.io import wavfile
import tables
import numbers
import glob
import random
import theano
import string
import theano.tensor as T
from theano.compat.python2x import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
# Sandbox?
import fnmatch
from theano.tensor.shared_randomstreams import RandomStreams


def labels_to_chars(labels):
    return "".join([chr(l + 97) for l in labels])


def _make_ctc_labels(y):
    # Assume that class values are sequential! and start from 0
    highest_class = np.max([np.max(d) for d in y])
    # Need to insert blanks at start, end, and between each label
    # See A. Graves "Supervised Sequence Labelling with Recurrent Neural
    # Networks" figure 7.2 (pg. 58)
    # (http://www.cs.toronto.edu/~graves/preprint.pdf)
    blank = highest_class + 1
    y_fixed = [blank * np.ones(2 * yi.shape[0] + 1).astype('int32')
               for yi in y]
    for i, yi in enumerate(y):
        y_fixed[i][1:-1:2] = yi
    return y_fixed


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
    return theano.shared(value=(0.01 * (rng.rand(*shape) - 0.5)).astype(
        theano.config.floatX), borrow=True)


def np_rand(shape, rng):
    return (0.01 * (rng.rand(*shape) - 0.5)).astype(theano.config.floatX)


def np_randn(shape, rng, name=None):
    """ Builds a numpy variable filled with random normal values """
    return (0.01 * rng.randn(*shape)).astype(theano.config.floatX)


def np_ortho(shape, rng, name=None):
    """ Builds a theano variable filled with orthonormal random values """
    g = rng.randn(*shape)
    o_g = linalg.svd(g)[0]
    return o_g.astype(theano.config.floatX)


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
            url = 'https://dl.dropboxusercontent.com/u/15378192/scribe2.pkl'
        except AttributeError:
            import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/scribe3.pkl'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    with open(data_file, 'rb') as pkl_file:
        data = cPickle.load(pkl_file)

    data_x, data_y = [], []
    for x, y in zip(data['x'], data['y']):
        data_y.append(np.asarray(y, dtype=np.int32))
        data_x.append(np.asarray(x, dtype=theano.config.floatX).T)

    train_x = data_x[:750]
    train_y = data_y[:750]
    valid_x = data_x[750:900]
    valid_y = data_y[750:900]
    test_x = data_x[900:]
    test_y = data_y[900:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


# A tricky trick for monkeypatching an instancemethod that is
# CPython :( there must be a better way
class _cVLArray(tables.VLArray):
    pass


def load_fruitspeech():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'audio.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if not os.path.isfile(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        except AttributeError:
            import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    if not os.path.exists(os.path.join(data_path, "audio")):
        tar = tarfile.open(data_file)
        os.chdir(data_path)
        tar.extractall()
        tar.close()

    h5_file_path = os.path.join(data_path, "saved_fruit.h5")
    if not os.path.exists(h5_file_path):
        data_path = os.path.join(data_path, "audio")

        audio_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.wav'):
                audio_matches.append(os.path.join(root, filename))

        random.seed(1999)
        random.shuffle(audio_matches)

        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_x_shapes = h5_file.createVLArray(h5_file.root, 'data_x_shapes',
                                              tables.Int32Atom(shape=()),
                                              filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        for wav_path in audio_matches:
            # Convert chars to int classes
            word = wav_path.split(os.sep)[-1][:-6]
            chars = [ord(c) - 97 for c in word]
            data_y.append(np.array(chars, dtype='int32'))
            fs, d = wavfile.read(wav_path)
            # Preprocessing from A. Graves "Towards End-to-End Speech
            # Recognition"
            Pxx, _, _, _ = plt.specgram(d, NFFT=256, noverlap=128)
            data_x_shapes.append(np.array(Pxx.T.shape, dtype='int32'))
            data_x.append(Pxx.T.astype('float32').flatten())
        h5_file.close()

    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_x_shapes = h5_file.root.data_x_shapes
    data_y = h5_file.root.data_y
    # A dirty hack to only monkeypatch data_x
    data_x.__class__ = _cVLArray

    # override getter so that it gets reshaped to 2D when fetched
    old_getter = data_x.__getitem__

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            return old_getter(key).reshape(data_x_shapes[key]).astype(
                theano.config.floatX)
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            return [o.reshape(s) for o, s in zip(
                self.read(start, stop, step), data_x_shapes[slice(
                    start, stop, step)])]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _cVLArray.__getitem__ = getter

    train_x = data_x[:80]
    train_y = data_y[:80]
    valid_x = data_x[80:90]
    valid_y = data_y[80:90]
    test_x = data_x[90:]
    test_y = data_y[90:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_cmuarctic():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    urls = ['http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_awb_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_clb_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_jmk_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_ksp_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_rms_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2',
            ]

    data_files = []

    for url in urls:
        dataset = url.split('/')[-1]
        data_file = os.path.join(data_path, dataset)
        data_files.append(data_file)
        if os.path.isfile(data_file):
            dataset = data_file
        if not os.path.isfile(data_file):
            try:
                import urllib
                urllib.urlretrieve('http://google.com')
            except AttributeError:
                import urllib.request as urllib
            print('Downloading data from %s' % url)
            urllib.urlretrieve(url, data_file)

    print('... loading data')

    folder_paths = []
    for data_file in data_files:
        folder_name = data_file.split(os.sep)[-1].split("-")[0]
        folder_path = os.path.join(data_path, folder_name)
        folder_paths.append(folder_path)
        if not os.path.exists(folder_path):
            tar = tarfile.open(data_file)
            os.chdir(data_path)
            tar.extractall()
            tar.close()

    h5_file_path = os.path.join(data_path, "saved_cmu.h5")
    if not os.path.exists(h5_file_path):
        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_x_shapes = h5_file.createVLArray(h5_file.root, 'data_x_shapes',
                                              tables.Int32Atom(shape=()),
                                              filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_meta = h5_file.createVLArray(h5_file.root, 'data_meta',
                                          tables.StringAtom(200),
                                          filters=tables.Filters(1))
        for folder_path in folder_paths:
            audio_matches = []
            for root, dirnames, filenames in os.walk(folder_path):
                for filename in fnmatch.filter(filenames, '*.wav'):
                    audio_matches.append(os.path.join(root, filename))

            f = open(os.path.join(folder_path, "etc", "txt.done.data"))
            read_raw_text = f.readlines()
            f.close()
            # Remove all punctuations
            list_text = [t.strip().lower().translate(
                string.maketrans("", ""), string.punctuation).split(" ")[1:-1]
                for t in read_raw_text]
            # Get rid of numbers, even though it will probably hurt
            # recognition on certain examples
            cleaned_lookup = {lt[0]: " ".join(lt[1:]).translate(
                None, string.digits).strip() for lt in list_text}
            data_meta.append(folder_path.split(os.sep)[-1])

            for wav_path in audio_matches:
                lookup_key = wav_path.split(os.sep)[-1][:-4]
                # Some files aren't consistent!
                if "_" in cleaned_lookup.keys()[0] and "_" not in lookup_key:
                    # Needs an _ to match text format... sometimes!
                    lookup_key = lookup_key[:6] + "_" + lookup_key[6:]
                elif "_" not in cleaned_lookup.keys()[0]:
                    lookup_key = lookup_key.translate(None, "_")
                try:
                    words = cleaned_lookup[lookup_key]
                    # Convert chars to int classes
                    chars = [ord(c) - 97 for c in words]
                    # Make spaces last class
                    chars = [c if c >= 0 else 26 for c in chars]
                    data_y.append(np.array(chars, dtype='int32'))
                    # Convert chars to int classes
                    fs, d = wavfile.read(wav_path)
                    # Preprocessing from A. Graves "Towards End-to-End Speech
                    # Recognition"
                    Pxx, _, _, _ = plt.specgram(d, NFFT=256, noverlap=128)
                    data_x_shapes.append(np.array(Pxx.T.shape, dtype='int32'))
                    data_x.append(Pxx.T.astype('float32').flatten())
                except KeyError:
                    # Necessary because some labels are missing in some folders
                    print("Skipping %s due to missing key" % wav_path)

        h5_file.close()

    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_x_shapes = h5_file.root.data_x_shapes
    data_y = h5_file.root.data_y
    # A dirty hack to only monkeypatch data_x
    data_x.__class__ = _cVLArray

    # override getter so that it gets reshaped to 2D when fetched
    old_getter = data_x.__getitem__

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            return old_getter(key).reshape(data_x_shapes[key]).astype(
                theano.config.floatX)
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            return [o.reshape(s) for o, s in zip(
                self.read(start, stop, step), data_x_shapes[slice(
                    start, stop, step)])]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _cVLArray.__getitem__ = getter

    train_x = data_x[:6000]
    train_y = data_y[:6000]
    valid_x = data_x[6000:7500]
    valid_y = data_y[6000:7500]
    test_x = data_x[7500:]
    test_y = data_y[7500:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_librispeech():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'dev-clean.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if not os.path.isfile(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
            url = 'http://www.openslr.org/resources/12/dev-clean.tar.gz'
        except AttributeError:
            import urllib.request as urllib
            url = 'http://www.openslr.org/resources/12/dev-clean.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    if not os.path.exists(os.path.join(data_path, "LibriSpeech", "dev-clean")):
        tar = tarfile.open(data_file)
        os.chdir(data_path)
        tar.extractall()
        tar.close()

    h5_file_path = os.path.join(data_path, "saved_libri.h5")
    if not os.path.exists(h5_file_path):
        data_path = os.path.join(data_path, "LibriSpeech", "dev-clean")

        audio_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.flac'):
                audio_matches.append(os.path.join(root, filename))

        text_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.txt'):
                text_matches.append(os.path.join(root, filename))

        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_x_shapes = h5_file.createVLArray(h5_file.root, 'data_x_shapes',
                                              tables.Int32Atom(shape=()),
                                              filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        for full_t in text_matches:
            f = open(full_t, 'r')
            for line in f.readlines():
                word_splits = line.strip().split(" ")
                file_tag = word_splits[0]
                words = word_splits[1:]
                # Convert chars to int classes
                chars = [ord(c) - 97 for c in (" ").join(words).lower()]
                # Make spaces last class
                chars = [c if c >= 0 else 26 for c in chars]
                data_y.append(np.array(chars, dtype='int32'))
                audio_path = [a for a in audio_matches if file_tag in a]
                if len(audio_path) != 1:
                    raise ValueError("More than one match for"
                                     "tag %s!" % file_tag)
                if not os.path.exists(audio_path[0][:-5] + ".wav"):
                    r = os.system("ffmpeg -i %s %s.wav" % (audio_path[0],
                                                           audio_path[0][:-5]))
                    if r:
                        raise ValueError("A problem occured converting flac to"
                                         "wav, make sure ffmpeg is installed")
                wav_path = audio_path[0][:-5] + '.wav'
                fs, d = wavfile.read(wav_path)
                # Preprocessing from A. Graves "Towards End-to-End Speech
                # Recognition"
                Pxx, _, _, _ = plt.specgram(d, NFFT=256, noverlap=128)
                data_x_shapes.append(np.array(Pxx.T.shape, dtype='int32'))
                data_x.append(Pxx.T.astype('float32').flatten())
            f.close()
        h5_file.close()

    h5_file_path = os.path.join(data_path, "saved_libri.h5")
    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_x_shapes = h5_file.root.data_x_shapes
    data_y = h5_file.root.data_y
    # A dirty hack to only monkeypatch data_x
    data_x.__class__ = _cVLArray

    # override getter so that it gets reshaped to 2D when fetched
    old_getter = data_x.__getitem__

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            return old_getter(key).reshape(data_x_shapes[key]).astype(
                theano.config.floatX)
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            return [o.reshape(s) for o, s in zip(
                self.read(start, stop, step), data_x_shapes[slice(
                    start, stop, step)])]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _cVLArray.__getitem__ = getter

    train_x = data_x[:2000]
    train_y = data_y[:2000]
    valid_x = data_x[2000:2500]
    valid_y = data_y[2000:2500]
    test_x = data_x[2500:]
    test_y = data_y[2500:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


class BaseNet(object):
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
    def get_sgd_updates(self, X_sym, y_sym, params, cost, learning_rate,
                        momentum):
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        for n, (param, gparam) in enumerate(zip(params, gparams)):
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * gparam
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step

        return updates

    def _norm_constraint(self, param, update_step, max_col_norm):
        stepped_param = param + update_step
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, max_col_norm)
            scale = desired_norms / (1e-7 + col_norms)
            new_param = param * scale
            new_update_step = update_step * scale
        else:
            new_param = param
            new_update_step = update_step
        return new_param, new_update_step

    def get_clip_sgd_updates(self, X_sym, y_sym, params, cost, learning_rate,
                             momentum, max_col_norm):
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        # Gradient clipping
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = 1.
        scaling_den = T.maximum(1., grad_norm)
        for n, (param, gparam) in enumerate(zip(params, gparams)):
            # clip gradient directly, not momentum etc.
            gparam = T.switch(not_finite, 0.1 * param,
                              gparam * (scaling_num / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * gparam
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates

    def get_clip_rmsprop_updates(self, X_sym, y_sym, params, cost,
                                 learning_rate, momentum, max_col_norm):
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "running_average_"):
            self.running_square_ = [0.] * len(gparams)
            self.running_avg_ = [0.] * len(gparams)
            self.updates_storage_ = [0.] * len(gparams)

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        # Gradient clipping
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = 1.
        scaling_den = T.maximum(1., grad_norm)
        for n, (param, gparam) in enumerate(zip(params, gparams)):
            gparam = T.switch(not_finite, 0.1 * param,
                              gparam * (scaling_num / scaling_den))
            combination_coeff = 0.9
            minimum_grad = 1e-4
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(gparam)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * gparam
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * (
                gparam / rms_grad)
            self.running_square_[n] = new_square
            self.running_avg_[n] = new_avg
            self.updates_storage_[n] = update_step
            self.momentum_velocity_[n] = update_step
            new_param, new_update_step = self._norm_constraint(
                param, update_step, max_col_norm)
            updates[param] = new_param + new_update_step

        return updates

    def get_sfg_updates(self, X_sym, y_sym, params, cost,
                        learning_rate, momentum, max_col_norm):
        gparams = T.grad(cost, params)
        updates = OrderedDict()
        from sfg import SFG
        if not hasattr(self, "sfg_"):
            self.count_ = theano.shared(0)
            self.slow_freq_ = 20
            self.sfg_ = SFG(params, gparams)

        slow_updates, fast_updates = self.sfg_.updates(self.learning_rate,
                                                       self.momentum,
                                                       epsilon=0.0001,
                                                       momentum_clipping=None)
        for param in slow_updates.keys():
            updates[param] = theano.ifelse.ifelse(T.eq(self.count_,
                                                       self.slow_freq_ - 1),
                                                  slow_updates[param],
                                                  fast_updates[param])
        updates[self.count_] = T.mod(self.count_ + 1, self.slow_freq_)
        return updates


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


class FeedforwardNetwork(BaseNet, TrainingMixin):
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

        self.params_ = params

        if self.learning_alg == "sgd":
            updates = self.get_sgd_updats(X_sym, y_sym, params, cost,
                                          self.learning_rate,
                                          self.momentum)
        else:
            raise ValueError("Algorithm %s is not "
                             "a valid argument for learning_alg!"
                             % self.learning_alg)
        self.fit_function = theano.function(
            inputs=[X_sym, y_sym], outputs=cost, updates=updates)
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


def _recurrent_tanh_init(input_size, hidden_size, output_size, random_state):
    wih = shared_rand((input_size, hidden_size), random_state)
    whh = shared_ortho((hidden_size, hidden_size), random_state)
    bh = shared_zeros((hidden_size,))
    h0 = shared_zeros((hidden_size,))
    params = [wih, bh, whh, h0]

    def step(x_t, h_tm1):
        h_t = T.tanh(T.dot(h_tm1, whh) + T.dot(x_t, wih) + bh)
        return h_t

    return step, params, [h0]


def build_recurrent_tanh_layer(input_size, hidden_size, output_size,
                               input_variable, random_state):
    step, params, outputs = _recurrent_tanh_init(input_size, hidden_size,
                                                 output_size, random_state)

    hidden, _ = theano.scan(
        step,
        sequences=[input_variable],
        outputs_info=outputs
    )
    return hidden, params


def _recurrent_relu_init(input_size, hidden_size, output_size, random_state):
    wih = shared_rand((input_size, hidden_size), random_state)
    whh = shared_ortho((hidden_size, hidden_size), random_state)
    bh = shared_zeros((hidden_size,))
    h0 = shared_zeros((hidden_size,))
    params = [wih, bh, whh, h0]

    def step(x_t, h_tm1):
        h_t = clip_relu(T.dot(h_tm1, whh) + T.dot(x_t, wih) + bh)
        return h_t

    return step, params, [h0]


def build_recurrent_relu_layer(input_size, hidden_size, output_size,
                               input_variable, random_state):

    step, params, outputs = _recurrent_relu_init(input_size, hidden_size,
                                                 output_size, random_state)
    hidden, _ = theano.scan(
        step,
        sequences=[input_variable],
        outputs_info=outputs
    )
    return hidden, params


def build_recurrent_lstm_layer(input_size, hidden_size, output_size,
                               input_variable, random_state, debug_step=False):
    # input to LSTM
    W_ = np.concatenate(
        [np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state)],
        axis=1)

    W = theano.shared(W_, borrow=True)

    # LSTM to LSTM
    U_ = np.concatenate(
        [np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state)],
        axis=1)

    U = theano.shared(U_, borrow=True)

    # bias to LSTM
    b = shared_zeros((4 * hidden_size,))

    # TODO: Ilya init for biases...
    params = [W, U, b]

    n_steps = input_variable.shape[0]
    n_features = input_variable.shape[1]

    def _slice(X, n, hidden_size):
        # Function is needed because tensor size changes across calls to step?
        if X.ndim == 3:
            return X[:, :, n * hidden_size:(n + 1) * hidden_size]
        return X[:, n * hidden_size:(n + 1) * hidden_size]

    def step(x_t, h_tm1, c_tm1):
        preactivation = T.dot(h_tm1, U)
        preactivation += x_t
        preactivation += b

        i_t = T.nnet.sigmoid(_slice(preactivation, 0, hidden_size))
        f_t = T.nnet.sigmoid(_slice(preactivation, 1, hidden_size))
        o_t = T.nnet.sigmoid(_slice(preactivation, 2, hidden_size))
        c_t = T.tanh(_slice(preactivation, 3, hidden_size))

        c_t = f_t * c_tm1 + i_t * c_t
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t, i_t, f_t, o_t, preactivation

    # Scan cannot handle batch sizes of 1?
    # Unbroadcast can fix it... but still weird
    #https://github.com/Theano/Theano/issues/1772
    init_hidden = T.zeros((1, hidden_size))
    init_hidden = T.unbroadcast(init_hidden, 0)
    init_cell = T.zeros((1, hidden_size))
    init_cell = T.unbroadcast(init_cell, 0)

    x = T.dot(input_variable, W) + b
    if debug_step:
        rval = step(x, init_hidden, init_cell)
    else:
        rval, _ = theano.scan(step,
                              sequences=[x,],
                              outputs_info=[init_hidden, init_cell,
                                            None, None, None, None],
                              n_steps=n_steps)

    hidden = rval[0]
    return hidden, params


def recurrence_relation(size):
    """
    Based on code from Shawn Tan
    """
    eye2 = T.eye(size + 2)
    return T.eye(size) + eye2[2:, 1:-1] + eye2[2:, :-2] * (T.arange(size) % 2)


def path_probs(predict, y_sym):
    """
    Based on code from Shawn Tan
    """
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


def _epslog(X):
    return T.cast(T.log(T.clip(X, 1E-12, 1E12)), theano.config.floatX)


def log_path_probs(y_hat_sym, y_sym):
    """
    Based on code from Shawn Tan with calculations in log space
    """
    pred_y = y_hat_sym[:, y_sym]
    rr = recurrence_relation(y_sym.shape[0])

    def step(logp_curr, logp_prev):
        return logp_curr + _epslog(T.dot(T.exp(logp_prev), rr))

    log_probs, _ = theano.scan(
        step,
        sequences=[_epslog(pred_y)],
        outputs_info=[_epslog(T.eye(y_sym.shape[0])[0])]
    )
    return log_probs


def ctc_cost(y_hat_sym, y_sym):
    """
    Based on code from Shawn Tan
    """
    forward_probs = path_probs(y_hat_sym, y_sym)
    backward_probs = path_probs(y_hat_sym[::-1], y_sym[::-1])[::-1, ::-1]
    probs = forward_probs * backward_probs / y_hat_sym[:, y_sym]
    total_probs = T.sum(probs)
    return -T.log(total_probs)


def log_ctc_cost(y_hat_sym, y_sym):
    """
    Based on code from Shawn Tan with sum calculations in log space
    """
    log_forward_probs = log_path_probs(y_hat_sym, y_sym)
    log_backward_probs = log_path_probs(
        y_hat_sym[::-1], y_sym[::-1])[::-1, ::-1]
    log_probs = log_forward_probs + log_backward_probs - _epslog(
        y_hat_sym[:, y_sym])
    log_probs = log_probs.flatten()
    max_log = T.max(log_probs)
    # Stable logsumexp
    loss = max_log + T.log(T.sum(T.exp(log_probs - max_log)))
    return -loss


def rnn_check_array(X, y=None):
    if type(X) == np.ndarray and len(X.shape) == 2:
        X = [X.astype(theano.config.floatX)]
    elif type(X) == np.ndarray and len(X.shape) == 3:
        X = X.astype(theano.config.floatX)
    elif type(X) == list:
        if type(X[0]) == np.ndarray and len(X[0].shape) == 2:
            X = [x.astype(theano.config.floatX) for x in X]
        else:
            raise ValueError("X must be a 2D numpy array or an"
                             "iterable of 2D numpy arrays")
    try:
        X[0].shape[1]
    except AttributeError:
        raise ValueError("X must be a 2D numpy array or an"
                         "iterable of 2D numpy arrays")

    if y is not None:
        if type(y) == np.ndarray and len(y.shape) == 1:
            y = [y.astype('int32')]
        elif type(y) == np.ndarray and len(y.shape) == 2:
            y = y.astype('int32')
        elif type(y) == list:
            if type(y[0]) == np.ndarray and len(y[0].shape) == 1:
                y = [yi.astype('int32') for yi in y]
            elif type(y[0]) != np.ndarray:
                y = [np.asarray(y).astype('int32')]
        try:
            y[0].shape[0]
        except AttributeError:
            raise ValueError("y must be an iterable of 1D numpy arrays")
        return X, y
    else:
        # If y is not passed don't return it
        return X


class RecurrentNetwork(BaseNet, TrainingMixin):
    def __init__(self, hidden_layer_sizes=[100], max_iter=1E2,
                 learning_rate=0.01, momentum=0., learning_alg="sgd",
                 recurrent_activation="tanh", max_col_norm=1.9365,
                 bidirectional=False, cost="ctc", save_frequency=10,
                 model_save_name="saved_model", random_seed=None,
                 input_checking=True):
        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.learning_rate = learning_rate
        self.learning_alg = learning_alg
        self.momentum = momentum
        self.bidirectional = bidirectional
        self.cost = cost
        self.max_col_norm = max_col_norm
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = int(max_iter)
        self.save_frequency = save_frequency
        self.model_save_name = model_save_name
        self.recurrent_activation = recurrent_activation
        self.input_checking = input_checking
        if recurrent_activation == "tanh":
            self.recurrent_function = build_recurrent_tanh_layer
        elif recurrent_activation == "relu":
            self.recurrent_function = build_recurrent_relu_layer
        elif recurrent_activation == "lstm":
            self.recurrent_function = build_recurrent_lstm_layer
        else:
            raise ValueError("Value %s not understood for recurrent_activation"
                             % recurrent_activation)

    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        input_variable = X_sym

        # layer_sizes consists of input size, all hidden sizes, and output size
        hidden_sizes = layer_sizes[1:-1]
        # set these to stop pep8 vim plugin from complaining
        input_size = None
        output_size = None
        for n in range(len(hidden_sizes)):
            if (n - 1) < 0:
                input_size = layer_sizes[0]
            else:
                input_size = output_size
            hidden_size = hidden_sizes[n]
            if (n + 1) != len(hidden_sizes):
                output_size = hidden_sizes[n + 1]
            else:
                output_size = layer_sizes[-1]

            forward_hidden, forward_params = self.recurrent_function(
                input_size, hidden_size, output_size, input_variable,
                self.random_state)

            if self.bidirectional:
                backward_hidden, backward_params = self.recurrent_function(
                    input_size, hidden_size, output_size, input_variable[::-1],
                    self.random_state)
                #Wfo = shared_rand((hidden_size, output_size), self.random_state)
                #Wbo = shared_rand((hidden_size, output_size), self.random_state)
                #by = shared_zeros((output_size,))
                #params = forward_params + backward_params + [Wfo, Wbo, by]
                #input_variable = T.dot(forward_hidden, Wfo) + T.dot(
                #    backward_hidden, Wbo) + by
                params = forward_params + backward_params
                input_variable = T.concatenate(
                    [forward_hidden, backward_hidden[::-1]],
                    axis=forward_hidden.ndim - 1)
            else:
                #Wo = shared_rand((hidden_size, output_size), self.random_state)
                #bo = shared_zeros((output_size,))
                #params = forward_params + [Wo, bo]
                #input_variable = T.dot(forward_hidden, Wo) + bo
                params = forward_params
                input_variable = forward_hidden

        # T.nnet.softmax doesn't define a gradient? wut
        #y_hat_sym = T.nnet.softmax(input_variable)
        # We can replace it with the mathematical expression and Theano will fix
        #y_hat_sym = T.exp(input_variable) / T.exp(input_variable).sum(
        #    1, keepdims=True)
        if self.bidirectional:
            sz = 2 * hidden_sizes[-1]
        else:
            sz = hidden_sizes[-1]
        Wo = shared_rand((sz, output_size),
                            self.random_state)
        bo = shared_zeros((output_size,))
        params = params + [Wo, bo]
        input_variable = T.dot(input_variable, Wo) + bo
        shp = input_variable.shape
        input_variable = input_variable.reshape([shp[0] * shp[1], shp[2]])
        y_hat_sym = T.nnet.softmax(input_variable)

        if self.cost == "ctc":
            cost = log_ctc_cost(y_hat_sym, y_sym)
        elif self.cost == "softmax":
            cost = softmax_cost(y_hat_sym, y_sym)

        self.params_ = params

        if self.learning_alg == "sgd":
            updates = self.get_clip_sgd_updates(
                X_sym, y_sym, params, cost, self.learning_rate, self.momentum,
                self.max_col_norm)
        elif self.learning_alg == "rmsprop":
            updates = self.get_clip_rmsprop_updates(
                X_sym, y_sym, params, cost, self.learning_rate, self.momentum,
                self.max_col_norm)
        elif self.learning_alg == "sfg":
            updates = self.get_sfg_updates(
                X_sym, y_sym, params, cost, self.learning_rate, self.momentum,
                self.max_col_norm)
        else:
            raise ValueError("Value of %s not a valid learning_alg!"
                             % self.learning_alg)

        self.fit_function = theano.function(inputs=[X_sym, y_sym],
                                            outputs=cost,
                                            updates=updates)

        self.loss_function = theano.function(inputs=[X_sym, y_sym],
                                             outputs=cost)

        self.predict_function = theano.function(
            inputs=[X_sym],
            outputs=[y_hat_sym],)

    def fit(self, X, y, valid_X=None, valid_y=None):
        if self.input_checking:
            X, y = rnn_check_array(X, y)
        input_size = X[0].shape[1]
        # Assume that class values are sequential! and start from 0
        highest_class = np.max([np.max(d) for d in y])
        lowest_class = np.min([np.min(d) for d in y])
        if lowest_class != 0:
            raise ValueError("Labels must start from 0!")
        if self.cost == "ctc":
            if self.input_checking:
                y = _make_ctc_labels(y)
            highest_class = np.max([np.max(d) for d in y])
        # Create a list of all classes, then get uniques
        # sum(lists, []) is list concatenation
        all_classes = np.unique(sum([list(np.unique(d)) for d in y], []))
        # +1 to include endpoint
        output_size = len(np.arange(lowest_class, highest_class + 1))
        X_sym = T.matrix('x')
        y_sym = T.ivector('y')
        #X_sym = T.tensor3('x')
        #y_sym = T.imatrix('y')
        self.layers_ = []
        self.layer_sizes_ = [input_size]
        self.layer_sizes_.extend(self.hidden_layer_sizes)
        self.layer_sizes_.append(output_size)
        if not hasattr(self, 'fit_function'):
            self._setup_functions(X_sym, y_sym,
                                  self.layer_sizes_)
        self.training_loss_ = []
        if valid_X is not None:
            if self.input_checking:
                valid_X, valid_y = rnn_check_array(valid_X, valid_y)
            self.validation_loss_ = []
            if self.input_checking:
                if self.cost == "ctc":
                    valid_y = _make_ctc_labels(valid_y)
                for vy in valid_y:
                    if not np.in1d(np.unique(vy), all_classes).all():
                        raise ValueError(
                            "Validation set contains classes not in training"
                            "set! Training set classes: %s\n, Validation set \
                             classes: %s" % (all_classes, np.unique(vy)))

        best_valid_loss = np.inf
        for itr in range(self.max_iter):
            print("Starting pass %d through the dataset" % itr)
            total_train_loss = 0
            for n in range(len(X)):
                X_n = X[n]
                y_n = y[n]
                train_loss = self.fit_function(X_n, y_n)
                total_train_loss += train_loss
            current_train_loss = total_train_loss / len(X)
            print("Training loss %f" % current_train_loss)
            self.training_loss_.append(current_train_loss)

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
        predictions = []
        if self.cost == "ctc":
            blank_pred = self.layer_sizes_[-1] - 1
            for n in range(len(X)):
                pred = np.argmax(self.predict_function(X[n])[0], axis=1)
                out = []
                for n, p in enumerate(pred):
                    if (p != blank_pred) and (n == 0 or p != pred[n-1]):
                        out.append(p)
                predictions.append(np.array(out))
            return predictions
        elif self.cost == "softmax":
            for n in range(len(X)):
                pred = np.argmax(self.predict_function(X[n])[0], axis=1)
                predictions.append(pred)
            return predictions

    def predict_proba(self, X):
        X = rnn_check_array(X)
        predictions = []
        for n in range(len(X)):
            pred = self.predict_function(X[n])[0]
            predictions.append(pred)
        return predictions
