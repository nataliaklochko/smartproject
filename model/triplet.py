import numpy as np

from keras.models import model_from_json
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from model.custom_model import CustomModel


class TripletModel(CustomModel):

    def __init__(self, name="Triplet_model"):
        super(TripletModel, self).__init__(name)

        model_path, weights_path = self._load_best_weights()

        with open(model_path, "r") as file:
            loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(filepath=weights_path)
        self.model = Model(
            outputs=model.layers[3].get_output_at(0),
            inputs=model.layers[3].get_input_at(0)
        )

    def predict(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return self.model.predict(x)[0]
        except Exception as exception:
            print("TripletModel couldn't predict: {0}".format(exception.args[0]))
            return []

    @staticmethod
    def _load_best_weights():
        """
        Chooses and loads weights with best score

        :return
        model_path: path to model architecture
        weights_path:

        """
        # TODO записывать скор в названия чекпоинтов при обучении модели, здесь парсить
        model_path = None
        weights_path = None
        return model_path, weights_path

