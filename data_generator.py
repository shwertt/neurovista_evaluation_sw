# %% import statements

import tensorflow as tf
import numpy as np
import scipy.io as io
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import h5py
from scipy.signal import resample


# define different pipeline stages
def data_loader(loader_args):
    # raise ValueError
    fname, label, standardize, shape, segment_length_minutes = loader_args
    if segment_length_minutes == 10:
        m = io.loadmat(fname)
        try:
            x = m['data']  # this is for the ecosystem data
        except KeyError:
            try:
                x = m['Data']  # this seems to be the continuous data
            except KeyError:
                x = m['dataStruct'][0][0][0]  # I believe this was the structure for the original contest data

    elif segment_length_minutes == 1:
        with h5py.File(fname, 'r') as f:
            x = np.array(f['Data'])
        x = x.T

    else:
        raise ValueError

    x = np.nan_to_num(x, copy=False)
    x = resample(x, 120000, axis=0)

    if standardize:
        if standardize.startswith('sm'):
            x -= x.mean(axis=0)
        if standardize.endswith('file'):
            x -= x.mean()
            std = x.std()
            if std:
                x /= std
        elif standardize.endswith('file_channelwise'):
            x -= x.mean(axis=0)
            std = x.std(axis=0)
            x[:, std.astype(bool)] /= std[std.astype(bool)]

    # resegment to 15 s segments
    try:
        x = x.reshape(shape)
    except ValueError:
        # if sequence is sampled with odd frequency, last segment will overlap a little with second-last segment
        x = np.concatenate([x[:x.shape[0] // shape[1] * shape[1]], x[-shape[1]:]])
        x = x.reshape(shape)
    y = np.ones(x.shape[0]) * label
    return x.astype('float32'), y.astype('int16')


# define several possible standardization methods
STANDARDIZE_OPTIONS = ['sm', 'sm_file', 'sm_file_channelwise', 'file', 'file_channelwise', None]


# %% define Tensorflow Generator that provides examples and labels
class TrainingGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 df_filenames_csv,
                 segment_length_minutes,
                 buffer_length=4000,
                 batch_size=1,
                 shuffle=True,
                 standardize_mode=None,
                 n_workers=10,
                 class_weights='auto'):

        # --- pass arguments ---
        self.batch_size = batch_size
        self.segment_length_minutes = segment_length_minutes
        self.csv = df_filenames_csv
        self.segm_length = 3000
        self.n_channels = 16
        self.samples_per_file = self.segment_length_minutes * 4  # draw 15 s segments
        if standardize_mode not in STANDARDIZE_OPTIONS:
            raise ValueError("'standardize_mode' hast to be one of {}.".format(STANDARDIZE_OPTIONS))
        self.standardize = standardize_mode

        self.shape = (len(self.csv) * self.samples_per_file, self.segm_length, self.n_channels)
        self.shuffle = shuffle

        # --- setup class weights ---
        if class_weights == 'auto' and type(self) == TrainingGenerator:
            pi = self.csv['class'].sum()
            ii = len(self.csv) - pi
            self.class_weights = {0: 1, 1: ii / pi}

            print('Class weights were set automatically based on the ratio in the training data')
            print('II: {}, PI: {}'.format(self.class_weights[0], self.class_weights[1]))

        elif class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = None

        # --- setup file IO ---
        # load a specified number of files, and enqueue them.
        # Draw samples randomly from those files and enqueue new files.
        if buffer_length > len(self.csv) * self.samples_per_file:
            buffer_length = len(self.csv) * self.samples_per_file
        self.buffer_length = buffer_length

        if not shuffle and n_workers > 1:
            warnings.warn('n_workers > 1 may result in shuffeled data. n_workers set to 1.')
            n_workers = 1
        self.n_workers = n_workers
        self.buffer = None
        self.pool = None
        self.pool_result = None

    def setup_buffer(self):
        if self.shuffle:
            self.buffer = tf.queue.RandomShuffleQueue(capacity=self.buffer_length,
                                                      min_after_dequeue=0,
                                                      dtypes=[tf.float32, tf.int16],
                                                      # shapes=[(self.segm_length, self.n_channels, 1), ()])  # for 2D
                                                      shapes=[(self.segm_length, self.n_channels), ()])  # for 1D
        else:
            self.buffer = tf.queue.FIFOQueue(capacity=self.buffer_length,
                                             dtypes=[tf.float32, tf.int16],
                                             # shapes=[(self.segm_length, self.n_channels, 1), ()])  # for 2D
                                             shapes=[(self.segm_length, self.n_channels), ()])  # for 1D

    def setup_buffer_pipeline(self):
        self.pool = ThreadPoolExecutor(max_workers=self.n_workers)
        if self.shuffle:
            data_files = self.csv.sample(frac=1).reset_index(drop=True)
        else:
            data_files = self.csv
        labels = data_files['class'].astype('int16')
        fnames = data_files['image']
        # shapes = ((-1, self.segm_length, self.n_channels, 1) for i in range(len(fnames)))  # for 2D
        shapes = ((-1, self.segm_length, self.n_channels) for i in range(len(fnames)))  # 1D
        standardizes = (self.standardize for i in range(len(fnames)))
        segment_length_minutes = (self.segment_length_minutes for i in range(len(fnames)))

        loader_args = zip(fnames, labels, standardizes, shapes, segment_length_minutes)
        self.pool_result = [self.pool.submit(self.data_enqueuer, i) for i in loader_args]

    def data_enqueuer(self, args):
        data = data_loader(args)

        try:
            self.buffer.enqueue_many(data)
        except Exception as e:  # queue is closed: do nothing, wait for cleanup
            pass

    def data_loader_error_handler(self, err_msg):
        print(err_msg)
        raise RuntimeError

    def cleanup_buffer(self):
        if self.buffer:
            self.buffer.close(cancel_pending_enqueues=True)
            if self.buffer.size():
                _ = self.buffer.dequeue_up_to(self.buffer_length)  # flush queue
        if self.pool:
            self.pool.shutdown(wait=True)
        self.pool = None
        self.pool_result = None
        self.buffer = None

    def on_epoch_end(self):
        # check if we have run through the whole dataset
        assert self.buffer.size() == 0

        if self.pool:
            assert np.all([f.done() for f in self.pool_result])

        self.cleanup_buffer()

    def on_epoch_start(self):
        self.cleanup_buffer()
        self.setup_buffer()
        self.setup_buffer_pipeline()  # fill buffer

    def __len__(self):
        return int(np.ceil(len(self.csv) * self.samples_per_file / self.batch_size))

    def __getitem__(self, index):
        if index == 0:
            self.on_epoch_start()
        # wait until the buffer is filled before returning values, in order to ensure shuffeled batches
        while self.buffer.size() < self.buffer_length and not np.all([f.done() for f in self.pool_result]):
            time.sleep(0.3)

        # if all jobs are done and buffer is not closed already, close it
        if np.all([f.done() for f in self.pool_result]) and not self.buffer.is_closed():
            self.buffer.close(cancel_pending_enqueues=False)

        x, y = self.buffer.dequeue_up_to(self.batch_size)

        if index == len(self) - 1:
            self.on_epoch_end()

        if self.class_weights is None:
            return x, y
        else:
            sw = np.zeros(y.shape)
            sw[(y == 0).numpy()] = self.class_weights[0]
            sw[(y == 1).numpy()] = self.class_weights[1]
            return x, y, sw


class EvaluationGenerator(TrainingGenerator):

    def setup_buffer(self):
        pass

    def setup_buffer_pipeline(self):
        pass

    def on_epoch_end(self):
        pass

    def on_epoch_start(self):
        pass

    def __getitem__(self, item):
        fname = self.csv['image'].iloc[item]
        # shape = (-1, self.segm_length, self.n_channels, 1)  # for 2D
        shape = (-1, self.segm_length, self.n_channels)  # for 1D
        x, _ = data_loader((fname, 0, self.standardize, shape, self.segment_length_minutes))
        return x

# %% execute only if run as a script
if __name__ == '__main__':
    # %% provide inputs
    import pandas as pd
    fn = '/home/s4238870/code/neurovista_evaluation/CSV/contest_train_data_labels.csv'
    fn = pd.read_csv(fn)
    fn = fn[:10]
    sl = 10  # segment length in minutes
    test_normalize = True
    test_epoch = True
    test_data = True
    # %% instantiate SegmentGenerator
    bs = 40
    if test_normalize:
        for run, sm in enumerate(STANDARDIZE_OPTIONS):
            print('\n\n=== Standardization mode: {} ==='.format(sm))
            gen = TrainingGenerator(fn,
                                    sl,
                                    shuffle=True,
                                    standardize_mode=sm,
                                    batch_size=bs)
            k = []
            batch = np.empty(1)
            if run == 0:
                print('Generating 10 batches...')
            for i in range(len(gen)):
                batch = gen[i]
                k.append(batch)
            if run == 0:
                print('First Batch shape: {}'.format(k[0][0].shape))
                print('Last Batch shape: {}'.format(k[-1][0].shape))
            X = np.concatenate([x[0] for x in k], axis=0)
            if run == 0:
                print('Data shape: {}'.format(X.shape))
            X = X.squeeze()
            print('\n----channelwise----')
            print('STD: {}'.format(X.std(-2).mean(axis=tuple([i for i in range(X.ndim - 2)]))))
            print('MEAN: {}'.format(X.mean(-2).mean(axis=tuple([i for i in range(X.ndim - 2)]))))

            print('\n----total----')
            print('STD: {}'.format(X.std(-2).mean()))
            print('MEAN: {}'.format(X.mean()))

    if test_epoch:
        gen = TrainingGenerator(fn,
                                sl,
                                shuffle=True,
                                standardize_mode='file_channelwise',
                                batch_size=bs)
        for epoch in range(10):
            peek = gen[0]
            print('Epoch: {}'.format(epoch))
            for i in range(len(gen)):
                X = gen[i]

    if test_data:
        gen = TrainingGenerator(fn,
                                sl,
                                shuffle=False,
                                standardize_mode=None,
                                batch_size=bs)
        for i in range(len(gen)):
            x, y, z = gen[i]
            m = io.loadmat(gen.csv.iloc[i]['image'])['dataStruct'][0][0][0]
            assert np.array_equal(x.numpy().reshape(m.shape), m)

    print('test passed.')

