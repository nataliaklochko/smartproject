import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from database.data_base import DataBase


class Cleaner(object):

    def __init__(self, db_name="db.sqlite3", window=100, log_dir="./cleaner_log"):
        self.db = DataBase(db_name=db_name)
        self.window = window
        self.log_dir = log_dir
        try:
            os.stat(log_dir)
        except:
            os.mkdir(log_dir)

    def remove_duplicates_from_dir(self, dir_name):
        """
        Remove image duplicates from dir_name folder

        :param dir_name:
        :return:
        """

        print("Removing duplicates from {0}".format(dir_name))
        log_file_path = os.path.join(self.log_dir, "{0}.log".format(dir_name))

        img_dict = {}
        for f in tqdm(os.listdir(dir_name)):
            img_dict[f] = np.array(Image.open(os.path.join(dir_name, f)))

        for f in tqdm(os.listdir(dir_name)):
            n = int(f.split("_")[0])

            for d in os.listdir(dir_name):
                m = int(d.split("_")[0])
                diff = m - n
                if d != f and diff in range(self.window):
                    if np.array_equal(img_dict[f], img_dict[d]):
                        os.remove(os.path.join(dir_name, d))
                        print("{0} removed".format(os.path.join(dir_name, d)))

                        with open(log_file_path, "w+") as log_file:
                            log_file.write(d)

    def remove_duplicates_from_db(self, dir_name):
        """
        Remove image duplicates from database
        (from all the tables)
        image names file constructed on dir_name folder

        :param dir_name:
        :return:
        """

        log_file_path = os.path.join(self.log_dir, "{0}.log".format(dir_name))
        with open(log_file_path, "r") as log_file:
            img_names = log_file.readlines()

        table_names = self.db.get_table_names()

        for img_name in img_names:
            for table_name in table_names:
                self.db.delete_by_name(table_name=table_name, img_name=img_name)
                print("{0} removed from {1}".format(img_name, table_name))

