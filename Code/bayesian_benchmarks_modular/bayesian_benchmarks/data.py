# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import pandas
import logging
from datetime import datetime
from scipy.io import loadmat
import pickle
import shutil
import pandas as pd
import tensorflow as tf

from sklearn.decomposition import PCA
from urllib.request import urlopen
logging.getLogger().setLevel(logging.INFO)
import zipfile

from bayesian_benchmarks.paths import DATA_PATH, BASE_SEED

_ALL_REGRESSION_DATATSETS = {}

def add_regression(C):
    _ALL_REGRESSION_DATATSETS.update({C.name:C})
    return C


def normalize(X):
    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    return (X - X_mean) / X_std, X_mean, X_std


class Dataset(object):
    def __init__(self, split=0, prop=0.9):
        if self.needs_download:
            self.download()

        X_raw, Y_raw = self.read_data()
        X, Y = self.preprocess_data(X_raw, Y_raw)

        ind = np.arange(self.N)

        np.random.seed(BASE_SEED + split)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        self.X_train = X[ind[:n]]
        self.Y_train = Y[ind[:n]]

        self.X_test = X[ind[n:]]
        self.Y_test = Y[ind[n:]]

    @property
    def datadir(self):
        dir = os.path.join(DATA_PATH, self.name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir

    @property
    def datapath(self):
        filename = self.url.split('/')[-1]  # this is for the simple case with no zipped files
        #print(filename)
        #print(os.path.join(self.datadir, filename))
        return os.path.join(self.datadir, filename)

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        logging.info('donwloading {} data'.format(self.name))

        is_zipped = np.any([z in self.url for z in ['.gz', '.zip', '.tar']])

        if is_zipped:
            filename = os.path.join(self.datadir, self.url.split('/')[-1])
        else:
            filename = self.datapath

        with urlopen(self.url) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        if is_zipped:
            zip_ref = zipfile.ZipFile(filename, 'r')
            zip_ref.extractall(self.datadir)
            zip_ref.close()

            # os.remove(filename)

        logging.info('finished donwloading {} data'.format(self.name))

    def read_data(self):
        raise NotImplementedError

    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        Y, self.Y_mean, self.Y_std = normalize(Y)
        return X, Y


uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


@add_regression
class Concrete(Dataset):
    N, D, name = 1030, 8, 'concrete'
    url = uci_base_url + 'concrete/compressive/Concrete_Data.xls'

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)



@add_regression
class Power(Dataset):
    N, D, name = 9568, 4, 'power'
    url = uci_base_url + '00294/CCPP.zip'

    @property
    def datapath(self):
        return os.path.join(self.datadir, 'CCPP/Folds5x2_pp.xlsx')

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)

@add_regression
class Protein(Dataset):
    N, D, name = 45730, 9, 'protein'
    url = uci_base_url + '00265/CASP.csv'

    def read_data(self):
        data = pandas.read_csv(self.datapath).values
        return data[:, 1:], data[:, 0].reshape(-1, 1)


# Andrew Wilson's datasets
#https://drive.google.com/open?id=0BxWe_IuTnMFcYXhxdUNwRHBKTlU
class WilsonDataset(Dataset):
    @property
    def datapath(self):
        n = self.name[len('wilson_'):]
        return '{}/uci/{}/{}.mat'.format(DATA_PATH, n, n)

    def read_data(self):
        data = loadmat(self.datapath)['data']
        return data[:, :-1], data[:, -1, None]


@add_regression
class Wilson_airfoil(WilsonDataset):
    name, N, D = 'wilson_airfoil', 1503, 5



@add_regression
class Wilson_parkinsons(WilsonDataset):
    name, N, D = 'wilson_parkinsons', 5875, 20


@add_regression
class Wilson_kin40k(WilsonDataset):
    name, N, D = 'wilson_kin40k', 40000, 8


##########################

regression_datasets = list(_ALL_REGRESSION_DATATSETS.keys())
regression_datasets.sort()

def get_regression_data(name, *args, **kwargs):
    return _ALL_REGRESSION_DATATSETS[name](*args, **kwargs)
