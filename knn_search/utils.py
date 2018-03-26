import numpy as np

import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context
from pycuda.tools import clear_context_caches


def check_pairwise_arrays(X, Y, dtype=np.float32):
    """ 
    Проверка данных перед использованием функций 
    для нахождения расстояния

    :param X: array-like, shape=(n_samples_1, n_features)
    :param Y: array-like, shape=(n_samples_2, n_features)

    :return
    X: np.array, shape=(n_samples_a_1, n_features)
    Y: np.array, shape=(n_samples_a_2, n_features)
    указатель на X, если аргумент Y=None
    """

    if Y is X or Y is None:
        X = Y = np.asarray(X, dtype=dtype)
    else:
        X = np.asarray(X, dtype=dtype)
        Y = np.asarray(Y, dtype=dtype)

    if len(X.shape) < 2:
        raise ValueError("X is required to be at least two dimensional.")
    if len(Y.shape) < 2:
        raise ValueError("Y is required to be at least two dimensional.")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            """Incompatible dimension for X and Y matrices: 
            X.shape[1] == %d while Y.shape[1] == %d""" % (X.shape[1], Y.shape[1])
        )
    return X, Y


def compute_distances(metric, kernel_code, X, Y=None):
    """
    Расчет расстояния для всех пар векторов (строк) 
    матриц X и Y (Y=X, если Y=None)

    :param metric: str, название функции расстояния
    :param kernel_code: str, код __global__ функции Cuda
    :param X: array-like, shape=(n_samples_1, n_features)
    :param Y: array-like, shape=(n_samples_2, n_features)

    :return
    distances: array, shape=(n_samples_1, n_samples_2)

    """
    drv.init()
    ctx = make_default_context()

    MAX_THREADS_PER_BLOCK = drv.Device(0).get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
    BLOCK_SIZE = int(np.sqrt(MAX_THREADS_PER_BLOCK))

    dx, mx = divmod(X.shape[0], BLOCK_SIZE)
    dy, my = divmod(X.shape[1], BLOCK_SIZE)
    gdim = ((dx + (mx > 0)), (dy + (my > 0)))

    mod = SourceModule(kernel_code)
    func = mod.get_function(metric)

    distances = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
    func(drv.In(X), drv.In(Y), drv.Out(distances), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)

    func(drv.In(X), drv.In(Y), drv.Out(distances), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)

    ctx.pop()
    clear_context_caches()

    return distances
