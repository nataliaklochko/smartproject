import unittest
import numpy as np

from scipy.spatial.distance import euclidean as eucl
from scipy.stats import pearsonr as pears
from scipy.spatial.distance import cosine as cos

from knn_search.metrics import euclidean_distances, cosine_distances, pearson_correlation
from knn_search.knn import find_knn


class TestKnnSearch(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(1000, 512)
        self.y = np.random.rand(1, 512)

    def test_cosine_distances(self):
        d1 = np.zeros(shape=(self.X.shape[0], 1))
        for i in range(self.X.shape[0]):
            d1[i] = 1 - cos(self.X[i, :], self.y[0, :])
        d1.reshape((self.X.shape[0], 1))
        d2 = cosine_distances(self.X, self.y)
        assert d1.shape == d2.shape
        np.testing.assert_almost_equal(d1, d2, decimal=5)

    def test_euclidean_distances(self):
        d1 = np.zeros(shape=(self.X.shape[0], 1))
        for i in range(self.X.shape[0]):
            d1[i] = eucl(self.X[i, :], self.y[0, :])
        d1.reshape((self.X.shape[0], 1))
        d2 = euclidean_distances(self.X, self.y)
        assert d1.shape == d2.shape
        np.testing.assert_almost_equal(d1, d2, decimal=5)

    def test_pearson_correlation(self):
        d1 = np.zeros(shape=(self.X.shape[0], 1))
        for i in range(self.X.shape[0]):
            d1[i] = pears(self.X[i, :], self.y[0, :])[0]
        d1.reshape((self.X.shape[0], 1))
        d2 = pearson_correlation(self.X, self.y)
        assert d1.shape == d2.shape
        np.testing.assert_almost_equal(d1, d2, decimal=5)

    def test_find_knn(self):
        X = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [5, 5, 5]
        ])
        y = np.array([4, 4, 4]).reshape((1, 3))
        nn = find_knn(X, y, euclidean_distances, p="neg", k=3)
        assert np.array_equal(nn, [2, 1, 0])
