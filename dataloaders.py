import os
import numpy as np
import pandas as pd

from scipy.io import loadmat
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


def _read_img2tensor(fname, grayscale=False):
    img = load_img(fname, grayscale=grayscale)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)
    return x


def _read_mat2tensor(fname):
    x = loadmat(fname)['image'].astype(np.float32)
    x = x.reshape((1,) + x.shape + (1,))
    return x
    

class BaseDataLoader:

    def train_generator(self, batch_size=32, shuffle=True):
        return self._load_batch(self.train, batch_size, shuffle), int(np.ceil(len(self.train) / float(batch_size)))

    def val_generator(self, batch_size=32, shuffle=False):
        return self._load_batch(self.val, batch_size, shuffle), int(np.ceil(len(self.val) / float(batch_size)))

    def test_generator(self, batch_size=32, shuffle=False):
        return self._load_batch(self.test, batch_size, shuffle), int(np.ceil(len(self.test) / float(batch_size)))

    def lengths(self):
        return {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}


class VDPDataLoader(BaseDataLoader):

    def __init__(self, data_dir, random_state=42):
        self.data_dir = data_dir
        self.load_vdp = load_vdp
        self.refs_dir = os.path.join(data_dir, 'ref')
        self.dist_dir = os.path.join(data_dir, 'stim')
        self.pmap_dir = os.path.join(data_dir, 'vdp')
            
        self.data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        self.train, valtest = train_test_split(self.data, test_size=0.2, random_state=random_state)
        self.val, self.test = train_test_split(valtest, test_size=0.5, random_state=random_state)

    def _data_gen(self, data, shuffle):
        while True:
            if shuffle:
                data = data.sample(frac=1)
            for index, sample in data.iterrows():
                ref_fname = os.path.join(self.refs_dir, sample.Reference)
                dist_fname = os.path.join(self.dist_dir, sample.Distorted)
                ref = _read_img2tensor(ref_fname)
                dist = _read_img2tensor(dist_fname)
                pmap_fname = os.path.join(self.pmap_dir, str(sample.Map))
                pmap = _read_img2tensor(pmap_fname, grayscale=True)
                yield [ref, dist, pmap, sample.Q]

    def _load_batch(self, data, batch_size, shuffle):
        c = self._data_gen(data, shuffle)
        while True:
            refs, dists, pmaps, qs = zip(*map(next, [c,] * batch_size))
            refs = np.vstack(refs)
            dists = np.vstack(dists)
            qs = np.array(qs, dtype=np.float32).reshape(-1, 1) / 100.0
            pmaps = np.vstack(pmaps)
            yield [refs, dists], [pmaps, qs]


class QDataLoader(BaseDataLoader):

    def __init__(self, data_dir, random_state=42):
        self.data_dir = data_dir
        self.refs_dir = os.path.join(data_dir, 'ref')
        self.dist_dir = os.path.join(data_dir, 'stim')
            
        self.data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        self.train, valtest = train_test_split(self.data, test_size=0.2, random_state=random_state)
        self.val, self.test = train_test_split(valtest, test_size=0.5, random_state=random_state)

    def _data_gen(self, data, shuffle):
        while True:
            if shuffle:
                data = data.sample(frac=1)
            for index, sample in data.iterrows():
                ref_fname = os.path.join(self.refs_dir, sample.Reference)
                dist_fname = os.path.join(self.dist_dir, sample.Distorted)
                ref = _read_img2tensor(ref_fname)
                dist = _read_img2tensor(dist_fname)
                yield [ref, dist, sample.Q]

    def _load_batch(self, data, batch_size, shuffle):
        c = self._data_gen(data, shuffle)
        while True:
            refs, dists, qs = zip(*map(next, [c,] * batch_size))
            refs = np.vstack(refs)
            dists = np.vstack(dists)
            qs = np.array(qs, dtype=np.float32).reshape(-1, 1) / 100.0
            yield [refs, dists], qs


class DRIIMDataLoader(BaseDataLoader):

    def __init__(self, data_dir, random_state=42):
        self.data_dir = data_dir
        self.refs_dir = os.path.join(data_dir, 'ref')
        self.dist_dir = os.path.join(data_dir, 'stim')
        self.driim_dir = os.path.join(data_dir, 'driim')
            
        self.data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        self.train, valtest = train_test_split(self.data, test_size=0.2, random_state=random_state)
        self.val, self.test = train_test_split(valtest, test_size=0.5, random_state=random_state)

    def _data_gen(self, data, shuffle):
        while True:
            if shuffle:
                data = data.sample(frac=1)
            for index, sample in data.iterrows():
                ref_fname = os.path.join(self.refs_dir, sample.Reference)
                ref = _read_mat2tensor(ref_fname)
                
                dist_fname = os.path.join(self.dist_dir, sample.Distorted)
                dist = _read_mat2tensor(dist_fname)
                
                driim_fnames = (sample['{}_Map'.format(i)] for i in 'ALR')
                driim_fnames = (os.path.join(self.driim_dir, i) for i in driim_fnames)
                driim = [_read_img2tensor(i, grayscale=True) for i in driim_fnames]
                driim = np.concatenate(driim, axis=3) # (1, 512, 512, 3)
                
                p75 = sample[['{}_P75'.format(i) for i in 'ALR']].values.astype(np.float32) / 100.0
                p95 = sample[['{}_P95'.format(i) for i in 'ALR']].values.astype(np.float32) / 100.0
                yield [ref, dist, driim, p75, p95]

    def _load_batch(self, data, batch_size, shuffle):
        c = self._data_gen(data, shuffle)
        while True:
            data = zip(*map(next, [c,] * batch_size))
            data = (np.vstack(i) for i in data)
            refs, dists, driims, p75s, p95s = data
            yield [refs, dists], [driims, p75s, p95s]
            
            
if __name__ == '__main__':
    d = DRIIMDataLoader('data/driim/v1')
    test_gen, _ = d.test_generator(batch_size=10)
    for [refs, dists], [driims, p75s, p95s] in test_gen:
        print refs.shape, refs.min(), refs.max()
        print dists.shape, dists.min(), dists.max()
        print driims.shape, driims.min(), driims.max()
        print p75s.shape, p75s.min(), p75s.max()
        print p95s.shape, p95s.min(), p95s.max()
        break



