import numpy as np

from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json

from model.custom_model import CustomModel


class VGGBottleneckModelFineTuned(CustomModel):

    def __init__(self,
                 name="VGG_bottleneck_model_fine_tuned_0",
                 json_model="../training/vgg_model/model_0.json",
                 weights="../training/vgg_model/model_0_weights-improvement-427-0.23.hdf5"):
        super(VGGBottleneckModelFineTuned, self).__init__(name)

        with open(json_model, "r") as file:
            json_str = file.read()
        base_model = model_from_json(json_str)
        base_model.load_weights(weights)

        self.model = Model(
            outputs=base_model.layers[19].output,
            inputs=base_model.input
        )
        print(self.model.summary())

    def predict(self, img_path):
        # (512,)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return self.model.predict(x)[0]
        except Exception as exception:
            print("VGGBottleneckModelFineTuned couldn't predict: {0}".format(exception.args[0]))
            return []
