import numpy as np


from keras.applications.nasnet import NASNetMobile
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
from model.custom_model import CustomModel


class NASNetBottleneckModel(CustomModel):

    def __init__(self, name="NASNet_bottleneck_model"):
        super(NASNetBottleneckModel, self).__init__(name)

        base_model = NASNetMobile(
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
        # (1056,)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return self.model.predict(x)[0]
        except Exception as exception:
            print("NASNetBottleneckModel couldn't predict: {0}".format(exception.args[0]))
            return []
