import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from model.custom_model import CustomModel


class VGGBottleneckModel(CustomModel):

    def __init__(self, name="VGG_bottleneck_model"):
        super(VGGBottleneckModel, self).__init__(name)

        base_model = VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            pooling="avg",
            input_shape=(224, 224, 3)
        )

        self.model = Model(
            outputs=base_model.layers[-1].output,
            inputs=base_model.input
        )

    def predict(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return self.model.predict(x)[0]
        except Exception as exception:
            print("VGGBottleneckModel couldn't predict: {0}".format(exception.args[0]))
            return []
