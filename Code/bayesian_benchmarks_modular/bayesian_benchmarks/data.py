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
_ALL_CLASSIFICATION_DATATSETS = {}



def add_regression(C):
    _ALL_REGRESSION_DATATSETS.update({C.name:C})
    return C

def add_classficiation(C):
    _ALL_CLASSIFICATION_DATATSETS.update({C.name:C})
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


#@add_regression
class Concrete(Dataset):
    N, D, name = 1030, 8, 'concrete'
    url = uci_base_url + 'concrete/compressive/Concrete_Data.xls'

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)



#@add_regression
class Power(Dataset):
    N, D, name = 9568, 4, 'power'
    url = uci_base_url + '00294/CCPP.zip'

    @property
    def datapath(self):
        return os.path.join(self.datadir, 'CCPP/Folds5x2_pp.xlsx')

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


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


#@add_regression
class Wilson_airfoil(WilsonDataset):
    name, N, D = 'wilson_airfoil', 1503, 5



#@add_regression
class Wilson_parkinsons(WilsonDataset):
    name, N, D = 'wilson_parkinsons', 5875, 20


#@add_regression
class Wilson_kin40k(WilsonDataset):
    name, N, D = 'wilson_kin40k', 40000, 8

@add_regression
class Wilson_protein(WilsonDataset):
    name, N, D = 'wilson_protein', 45730, 9

##########################

regression_datasets = list(_ALL_REGRESSION_DATATSETS.keys())
regression_datasets.sort()

def get_regression_data(name, *args, **kwargs):
    return _ALL_REGRESSION_DATATSETS[name](*args, **kwargs)





class Classification(Dataset):
    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        return X, Y

    @property
    def needs_download(self):
        if os.path.isfile(os.path.join(DATA_PATH, 'classification_data', 'iris', 'iris_R.dat')):
            return False
        else:
            return True

    def download(self):
        logging.info('donwloading classification data. WARNING: downloading 195MB file'.format(self.name))

        filename = os.path.join(DATA_PATH, 'classification_data.tar.gz')

        url = 'http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz'
        with urlopen(url) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        import tarfile
        tar = tarfile.open(filename)
        tar.extractall(path=os.path.join(DATA_PATH, 'classification_data'))
        tar.close()

        logging.info('finished donwloading {} data'.format(self.name))


    def read_data(self, components = None):
        datapath = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_R.dat')
        if os.path.isfile(datapath):
            data = np.array(pandas.read_csv(datapath, header=0, delimiter='\t').values).astype(float)
        else:
            data_path1 = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_train_R.dat')
            data1 = np.array(pandas.read_csv(data_path1, header=0, delimiter='\t').values).astype(float)

            data_path2 = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_test_R.dat')
            data2 = np.array(pandas.read_csv(data_path2, header=0, delimiter='\t').values).astype(float)

            data = np.concatenate([data1, data2], 0)

        return data[:, :-1], data[:, -1].reshape(-1, 1)


rescale = lambda x, a, b: b[0] + (b[1] - b[0]) * x / (a[1] - a[0])






mnist_datasets = [['MNIST', 70000, 15, 10],
                ['FASHION_MNIST',70000,15,10]]

for name, N, D, K in mnist_datasets:
    @add_classficiation
    class MNIST(Classification):
        name, N, D, K = name, N, D, K

        def read_data(self, components = 15):
            """Reads data from tf.keras and concatenates training and testing sets for randomisation
            - Creates features as prinicipal components (PCA) of pixel values  
            - Output: (feature matrix: 700000 X n_components , labels vector(ordinal) : 70000 x 1 
            """
            
            if name == 'FASHION_MNIST':
                (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
            else:
                (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

            train_images = train_images / 255.0
            test_images = test_images / 255.0
            train_images = np.reshape(train_images, (60000, 784))
            test_images = np.reshape(test_images, (10000, 784))
            train_images = tf.concat([train_images, test_images], 0)
            train_labels = tf.concat([train_labels, test_labels], 0)
            #Uses sklearn PCA
            pca = PCA(n_components = components)
            pca.fit(train_images)
            train_image_pca = pca.transform(train_images)
            train_labels = tf.dtypes.cast(train_labels, tf.int64)
            train_labels = tf.reshape(train_labels, [-1, 1])

            return train_image_pca, train_labels.numpy()

        
classification_datasets = list(_ALL_CLASSIFICATION_DATATSETS.keys())
classification_datasets.sort()


def get_classification_data(name, *args, **kwargs):
    return _ALL_CLASSIFICATION_DATATSETS[name](*args, **kwargs)