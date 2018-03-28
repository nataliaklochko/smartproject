import os
import pickle
import numpy as np
from tensorflow import get_default_graph

from database.data_base import DataBase
from knn_search.knn import find_knn
from knn_search.metrics import pearson_correlation, euclidean_distances, cosine_distances
from sklearn.decomposition import PCA, KernelPCA


class ImageProcessing(object):

    def __init__(self, model, dims, find_pca=False):
        self.graph = get_default_graph()
        self.model = model()

        print("Loading dataset ...")
        self.db = DataBase()
        self.db.c.execute("SELECT ID, name, {0}  FROM smart_pot".format(self.model.name))
        self.data = self.db.c.fetchall()

        self.vectors = []
        self.indices = []
        self.names = []
        self.links = []

        for item in self.data:
            try:
                self.vectors.append(np.frombuffer(item[2], dtype=np.float32))
                self.indices.append(item[0] - 1)
                self.names.append(item[1])
                self.links.append("http://inhome360.ru/catalog/show/" + item[1].split("_")[2])
            except IndexError as err:
                print(err.args)
                print(item[1])

        print("Dataset was loaded.")

        pca_path = os.path.join(os.path.dirname(__file__), "utils",
                                "pca_{0}_{1}_to_{2}.pickle".format(self.model.name, dims[0], dims[1]))
        if find_pca:
            print("Fitting PCA...")
            self.pca = PCA(n_components=dims[1], copy=False)
            self.vectors = self.pca.fit_transform(self.vectors)
            with open(pca_path, "wb") as file:
                pickle.dump(self.pca, file)
        else:
            try:
                with open(pca_path, "rb") as file:
                    self.pca = pickle.load(file)
                self.vectors = self.pca.transform(self.vectors)
            except:
                pass

        print("Feature vectors are transformed")

    def get_features(self, img_path):
        with self.graph.as_default():
            prediction = self.model.predict(img_path).reshape((1, -1))
            try:
                prediction = self.pca.transform(prediction)
            except:
                pass
            feature_array = prediction.reshape((1, -1))
            return feature_array

    def find_similar(self, feature_vector, num_nearest):
        top_k_indices_id = []
        top_k_names = []
        top_k_links = []
        top_k_indices = find_knn(self.vectors, feature_vector, pearson_correlation, "pos", num_nearest)

        for i in top_k_indices:
            top_k_indices_id.append(self.indices[i])
            top_k_names.append(self.names[i])
            top_k_links.append(self.links[i])

        return top_k_indices_id, top_k_names, top_k_links

    def main(self, img_path, num_nearest=10):
        """
        :param img_path:  path to image to find similar of (in the media folder)

        :return
        sim_images: list of similar image names (from 'static' folder)

        """

        try:
            f_vec = self.get_features(img_path)
            sim_inds, sim_names, sim_links = self.find_similar(f_vec, num_nearest)
            return sim_names, sim_links
        except Exception as e:
            print(e.args)
            print("Не нашёл похожих...")
            return [], []

