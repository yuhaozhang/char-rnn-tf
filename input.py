import sys
import os
import random
import math
import collections
import cPickle
import numpy as np

class TextReader(object):

    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise IOError("Data file not found.")
        self.data_dir = os.path.dirname(data_path)
        self.data_path = data_path

    def build_vocab(self):
        with open(self.data_path, 'r') as infile:
            counter = collections.Counter(infile.read())
        chars, _ = zip(*counter.most_common())
        self.vocab_size = len(chars)
        self.char2id = dict(zip(chars, range(self.vocab_size)))
        _dump_to_file(self.char2id, os.path.join(self.data_dir, 'vocab.cPickle'))
        return self.char2id

    def convert_text_to_ids(self):
        with open(self.data_path, 'r') as infile:
            self.id_data = [self.char2id[c] for c in infile.read()]
        self.total_chars = len(self.id_data)
        # _dump_to_file(self.id_data, os.path.join(self.data_dir, 'char_data.cPickle'))

    def split_data(self, test_fraction):
        num_train = int(math.floor(self.total_chars * (1-test_fraction)))
        self.train_data = self.id_data[:num_train]
        self.test_data = self.id_data[num_train:]
        _dump_to_file(self.train_data, os.path.join(self.data_dir, 'train.cPickle'))
        _dump_to_file(self.test_data, os.path.join(self.data_dir, 'test.cPickle'))

    def prepare_data(self, test_fraction=0.05):
        self.build_vocab()
        self.convert_text_to_ids()
        self.split_data(test_fraction)

class DataLoader(object):

    def __init__(self, data_path, batch_size, num_steps):
        """
        For batch RNN training, we need to chunk the data into batch_size chunks. For each chunk,
        we need to further chunk it into num_steps batches. Thus, each batch is of size
        [batch_size, num_steps]
        """
        self.raw_data = _load_from_dump(data_path)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.total_chars = len(self.raw_data)

        self.raw_data = np.array(self.raw_data, dtype=np.int32)
        batch_len = self.total_chars // batch_size
        self.num_batch = (batch_len - 1) // num_steps
        data_array = np.zeros([batch_size, batch_len])
        for i in range(batch_size):
            data_array[i] = self.raw_data[i*batch_len:(i+1)*batch_len]

        self.x_list, self.y_list = [], []
        for i in range(self.num_batch):
            self.x_list.append(data_array[:, i*num_steps:(i+1)*num_steps])
            self.y_list.append(data_array[:, i*num_steps+1:(i+1)*num_steps+1])
        self.cur_pos = 0

    def next_batch(self):
        if self.cur_pos >= self.num_batch:
            self.cur_pos = 0
        self.cur_pos += 1
        return self.x_list[self.cur_pos-1], self.y_list[self.cur_pos-1]


def _dump_to_file(obj, filename):
    with open(filename, 'w') as outfile:
        cPickle.dump(obj, outfile)
    return

def _load_from_dump(filename):
    with open(filename, 'r') as infile:
        obj = cPickle.load(infile)
    return obj


def test():
    reader = TextReader('./data/tinyshakespeare/input.txt')
    reader.prepare_data()
    loader = DataLoader('./data/tinyshakespeare/train.cPickle', 128, 30)
    print loader.total_chars
    print loader.num_batch
    print loader.next_batch()

if __name__ == '__main__':
    test()