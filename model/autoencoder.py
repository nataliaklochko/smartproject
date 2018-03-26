import os
import numpy as np

from keras.layers import GlobalAveragePooling2D
from keras.models import load_model as load
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import tensorflow as tf

from model.custom_model import CustomModel


class AutoencoderModel(CustomModel):

    def __init__(self, name="Autoencoder_model"):
        super(AutoencoderModel, self).__init__(name)

        base_model = load(os.path.join("utils", "autoencoder_model_0.h5"))
        self.model = Model(base_model.input, GlobalAveragePooling2D()(base_model.layers[6].output))

    def predict(self, img_path):
        # (512,)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return self.model.predict(x)[0]
        except Exception as exception:
            print("AutoencoderModel couldn't predict: {0}".format(exception.args[0]))
            return []

