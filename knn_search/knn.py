import cupy


def find_knn(X, y, metric, p="pos", k=10):
    """
    Finds k Nearest Neibhors for y in X matrix (rows are vectors)
    due to metric

    :param X: matrix (n_samples, n_features)
    :param y: feature vector (1, n_features)
    :param metric: euclidean_distances, cosine_distances, pearson_correlation
    :param p: sort direction
            pos (для ошибки по убыванию pears, cos) or
            neg (для ошибки по возрастанию eucl) )
    :param k: the number of Nearest Neibhors (default=10)

    :return: k indices of Nearest Neibhours (list)

    """
    distances = metric(X, y)

    args = cupy.argsort(distances, axis=0)
    sorted_args = []

    if p == "pos":
        sorted_args = [i for i in args[-k:][::-1, 0]]
    elif p == "neg":
        sorted_args = [i for i in args[:k][:, 0]]
    else:
        print("Choose sort direction: pos or neg")

    return sorted_args

