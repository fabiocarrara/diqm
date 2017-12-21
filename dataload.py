import os
import numpy as np
import pandas as pd

from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


def _read_img2tensor(fname, grayscale=False):
    img = load_img(fname, grayscale=grayscale)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)
    return x


class DataLoader:

    def __init__(self, data_dir, random_state=42, load_vdp=True):
        self.data_dir = data_dir
        self.load_vdp = load_vdp
        self.refs_dir = os.path.join(data_dir, 'ref')
        self.dist_dir = os.path.join(data_dir, 'stim')
        if self.load_vdp:
            self.pmap_dir = os.path.join(data_dir, 'vdp')
            
        self.data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        self.train, valtest = train_test_split(self.data, test_size=0.2, random_state=random_state)
        self.val, self.test = train_test_split(valtest, test_size=0.5, random_state=random_state)

    def __data_gen(self, data, shuffle):
        while True:
            if shuffle:
                data = data.sample(frac=1)
            for index, sample in data.iterrows():
                ref_fname = os.path.join(self.refs_dir, sample.Reference)
                dist_fname = os.path.join(self.dist_dir, sample.Distorted)
                ref = _read_img2tensor(ref_fname)
                dist = _read_img2tensor(dist_fname)
                if self.load_vdp:
                    pmap_fname = os.path.join(self.pmap_dir, str(sample.Map))
                    pmap = _read_img2tensor(pmap_fname, grayscale=True)
                    yield [ref, dist, pmap, sample.Q]
                else:
                    yield [ref, dist, sample.Q]

    def __load_batch(self, data, batch_size, shuffle):
        c = self.__data_gen(data, shuffle)
        while True:
            XY = zip(*map(next, [c,] * batch_size))
            if self.load_vdp:
                refs, dists, pmaps, qs = XY
            else:
                refs, dists, qs = XY
                
            refs = np.vstack(refs)
            dists = np.vstack(dists)
            qs = np.array(qs, dtype=np.float32).reshape(-1, 1) / 100.0
            
            if self.load_vdp:            
                pmaps = np.vstack(pmaps)
                yield [refs, dists], [pmaps, qs]
            else:
                yield [refs, dists], qs

    def train_generator(self, batch_size=32, shuffle=True):
        return self.__load_batch(self.train, batch_size, shuffle), int(np.ceil(len(self.train) / float(batch_size)))

    def val_generator(self, batch_size=32, shuffle=False):
        return self.__load_batch(self.val, batch_size, shuffle), int(np.ceil(len(self.val) / float(batch_size)))

    def test_generator(self, batch_size=32, shuffle=False):
        return self.__load_batch(self.test, batch_size, shuffle), int(np.ceil(len(self.test) / float(batch_size)))

    def lengths(self):
        return {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}


