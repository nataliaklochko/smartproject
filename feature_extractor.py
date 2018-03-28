import os
import pickle
import numpy as np
from database.data_base import DataBase
from knn_search.knn import find_knn
from knn_search.metrics import pearson_correlation

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input


class FeatureExtractor(object):

    def __init__(self, model, imageset_dir, table_name="features_table", types=None, load_names=False):
        """
        Creates object to collect features database and image_dataset for training

        :param model: model to extract from
        :param imageset_dir: image dataset directory
        :param table_name: default - features_table
        :param types: default - None (list)
        :param load_names: default - False

        """
        self.model = model()
        self.db = DataBase()
        self.table_name = table_name
        self.imageset_dir = imageset_dir
        self.types = types
        self._setup(load_names)

    def _setup(self, load_names):
        """
        Creates table if not exists and column for model features

        """
        if not load_names:
            self.db.create_feature_table(self.table_name)
        elif self.types:
            names = []
            for t in self.types:
                for img_name in os.listdir(os.path.join(self.imageset_dir, t)):
                    names.append((img_name, int(self.types.index(t))))
            self.db.create_feature_table(table_name=self.table_name, img_names=names)
        else:
            names = []
            for img_name in os.listdir(self.imageset_dir):
                names.append((img_name, -1))
            self.db.create_feature_table(table_name=self.table_name, img_names=names)
        try:
            self.db.create_column(self.table_name, self.model.name)
        except:
            print("Column {0} exists".format(self.model.name))

    def extract(self):
        """
        Extract features and write to the table

        """
        self.model.load_dataset(
            dir_path=self.imageset_dir,
            db=self.db,
            table=self.table_name,
            types=None
        )

    def create_training_dataset(self):
        try:
            os.stat(os.path.join(self.imageset_dir, self.model.name))
        except:
            os.mkdir(os.path.join(self.imageset_dir, self.model.name))

        for n in range(len(self.types)):
            names, features = self.db.find_by_type(self.table_name, self.model.name, n)
            result = find_knn(X=features, y=None, metric=pearson_correlation, p="pos", k=100)
            result_dict = {}
            for k, v in zip(names, result):
                result_dict[k] = v
            file_path = os.path.join(self.imageset_dir, self.model.name, "{0}_{1}.pickle".format(n, "pears"))
            with open(file_path, "wb") as fp:
                pickle.dump(result_dict, fp)

    def _load_items(self):
        data = []
        for n in range(len(self.types)):
            file_path = os.path.join(self.imageset_dir, self.model.name, "{0}_{1}.pickle".format(n, "pears"))
            with open(file_path, "rb") as f:
                data.append(pickle.load(f))
        return data

    @staticmethod
    def _img_prep(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def _get_triplet(self, n, k, anchor_name, pos_n, neg_k):
        pos_name = self.db.find_name_by_id(self.table_name, pos_n)
        neg_name = self.db.find_name_by_id(self.table_name, neg_k)
        anchor = self._img_prep(os.path.join(self.imageset_dir , self.types[n], anchor_name))
        pos = self._img_prep(os.path.join(self.imageset_dir, self.types[n], pos_name))
        neg = self._img_prep(os.path.join(self.imageset_dir, self.types[k], neg_name))
        return (anchor, pos, neg)

    def triplet_batch_generator(self):
        batch_size = 32
        data = self._load_items()
        len_types = len(self.types.keys())
        size_sim = 50
        targets = np.ones(shape=(batch_size, 2, 1))

        while True:
            batch_anchor = []
            batch_pos = []
            batch_neg = []
            n = np.random.randint(0, len_types)
            k = np.random.randint(0, len_types)
            m = np.random.randint(0, size_sim)
            try:
                for _ in range(batch_size):
                    item = data[n].popitem()
                    _item = data[k].popitem()
                    (anchor, pos, neg) = self._get_triplet(n, k, item[0], item[1][m], _item[1][m])
                    batch_anchor.append(anchor)
                    batch_pos.append(pos)
                    batch_neg.append(neg)
                yield ([
                    np.array(batch_anchor).reshape((batch_size, 224, 224, 3)),
                    np.array(batch_pos).reshape((batch_size, 224, 224, 3)),
                    np.array(batch_neg).reshape((batch_size, 224, 224, 3))
                ],targets)

            except KeyError:
                data = self._load_items()
