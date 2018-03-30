import os
from tqdm import tqdm


class CustomModel(object):
    """
    Model abstraction
    """

    def __init__(self, name):
        """
        Creates new model

        :param name: model name (str)

        """
        self.name = name

    @classmethod
    def get_name(cls):
        return cls.__name__

    def predict(self, img_path):
        """
        Finds feature vector

        :param img_path: image path
        :return prediction: np.array(dtype=np.float32, shape=(n_features,))

        """
        raise NotImplementedError

    def add_new_img(self, db, table, dir_path, img_name):
        prediction = self.predict(img_path=os.path.join(dir_path, img_name))
        db.insert_image_features(table, self.name, img_name, prediction)

    def load_dataset(self, db, table, dir_path=None, names=None, types=None):
        """
        Loads image image_dataset to database
        Creates new column "name" in table and fills with feature vectors

        :param dir_path: path to image directory
        :param db: database (DataBase)
        :param table: table name (str)
        :param types: dir names in dir_path (list   )

        """
        if types:
            for t in types:
                self._load_dir(os.path.join(dir_path, t), db, table)
        else:
            self._load_dir(dir_path, db, table)

    def _load_dir(self, db, table, dir_path=None, names=None, types=None):
        """
        Creates new column "name" in table and fills with feature vectors

        :param dir_path: path to image directory
        :param db: database (DataBase)
        :param table: table name (str)

        """

        print("Exctracting features from {0}...".format(dir_path))

        if dir_path:
            for img_name in tqdm(os.listdir(dir_path)):
                    prediction = self.predict(img_path=os.path.join(dir_path, img_name))
                    db.insert_image_features(table, self.name, img_name, prediction)

        elif names:
            for img_name in tqdm(names):
                prediction = self.predict(img_path=os.path.join(dir_path, img_name))
                db.insert_image_features(table, self.name, img_name, prediction)

