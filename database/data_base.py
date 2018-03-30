import os
import sqlite3
import numpy as np


class DataBase(object):

    def __init__(self, db_name="db.sqlite3"):
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_name)
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

    def create_feature_table(self, table_name, img_names=None):
        """

        :param table_name:
        :param img_names: [(img_name_1, type_1), (img_name_2, type_2), ...]

        """
        self.c.execute("""
              CREATE TABLE IF NOT EXISTS {0} 
              (ID INTEGER PRIMARY KEY AUTOINCREMENT, 
              name TEXT,
              type INTEGER DEFAULT -1)
              """.format(table_name))
        self.conn.commit()
        if img_names:
            self.c.execute("SELECT name FROM {0}".format(table_name))
            names_loaded = self.c.fetchall()
            names_loaded = [n[0] for n in names_loaded]
            img_names = [n[0] for n in img_names]
            img_names = list(set(img_names) - set(names_loaded))
            img_names = [(n, -1) for n in img_names]
            self.c.executemany("INSERT INTO {0}(name, type) VALUES (?, ?)".format(table_name), img_names)
        self.conn.commit()

    def get_names(self):
        self.c.execute("SELECT name FROM smart_pot WHERE ID > 46854")
        names = [n[0] for n in self.c.fetchall()]
        return names

    def close_conn(self):
        self.c.close()
        self.conn.close()

    def create_column(self, table_name, column_name):
        self.c.execute("ALTER TABLE {0} ADD COLUMN '{1}' 'BLOB'".format(table_name, column_name))
        self.conn.commit()

    def insert_image_features(self, table_name, column_name, img_name, prediction):
        self.c.execute("UPDATE {0} SET {1}=? WHERE name=?".format(table_name, column_name), (prediction, img_name))
        self.conn.commit()

    def find_by_type(self, table_name, column_name, t):
        self.c.execute("SELECT name, {0} FROM {1} WHERE type=?".format(column_name, table_name), (int(t),))
        data = self.c.fetchall()
        names = []
        vects = []
        for d in data:
            names.append(d[0])
            vects.append(np.frombuffer(d[1], dtype=np.float32))
        return names, vects

    def find_name_by_id(self, table_name, id):
        self.c.execute("SELECT name FROM {1} WHERE ID=?".format(table_name), (int(id),))
        return self.c.fetchall()

    def write_type(self, table_name, img_name, t):
        self.c.execute("UPDATE {0} SET type=? WHERE name=?".format(table_name), (t, img_name))
        self.conn.commit()

    def delete_by_name(self, table_name, img_name):
        self.c.execute("DELETE FROM {0} WHERE name=?".format(table_name), (img_name,))
        self.conn.commit()

    def find_by_name(self, table_name, img_name):
        self.c.execute("SELECT id FROM {0} WHERE name=?".format(table_name), (img_name,))
        return self.c.fetchall()
