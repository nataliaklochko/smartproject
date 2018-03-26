import unittest
import numpy as np
from database.data_base import DataBase


class TestDb(unittest.TestCase):

    def setUp(self):
        self.db = DataBase()
        self.table_name = "test_table"
        self.c_name = "kek_col"

    def test_create_table(self):
        self.db.create_feature_table(table_name=self.table_name)
        self.db.c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [n[0] for n in self.db.c.fetchall()]
        assert self.table_name in table_names

    def test_insert_images(self):
        img_names = [("kek.png", 3), ("lol.jpg", 5), ("lal.jpg", 2)]
        self.db.create_feature_table(table_name=self.table_name, img_names=img_names)
        self.db.c.execute("SELECT name FROM {0}".format(self.table_name))
        outputs = [x[0] for x in self.db.c.fetchall()]
        img_names = [x[0] for x in img_names]
        assert set(outputs) == set(img_names)

    def test_create_column(self):
        self.db.create_column(table_name=self.table_name, column_name=self.c_name)
        cursor = self.db.c.execute("SELECT * FROM {0}".format(self.table_name))
        c_names = list(map(lambda x: x[0], cursor.description))
        assert self.c_name in c_names

    def test_insert_image_features(self):
        img_name = "kek.png"
        prediction = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.db.insert_image_features(
            table_name=self.table_name,
            column_name=self.c_name,
            img_name=img_name,
            prediction=prediction
        )
        self.db.c.execute("SELECT {0} FROM {1} WHERE name=?".format(self.c_name, self.table_name), (img_name,))
        result = np.frombuffer(self.db.c.fetchall()[0][0], dtype=np.float32)
        assert np.array_equal(prediction, result)
