import os
import pickle

from model.vgg_bottleneck import VGGBottleneckModel
from model.resnet_bottleneck import ResNetBottleneckModel
from model.densenet_bottleneck import DenseNetBottleneckModel
from model.nasnet_bottleneck import NASNetBottleneckModel
from model.vgg_bottleneck_fine_tuned import VGGBottleneckModelFineTuned
from model.autoencoder import AutoencoderModel

from feature_extractor import FeatureExtractor


media_dir = os.path.join("..", "inhome360")


types = [
    "free_standing_lights",
    "chandeliers",
    "mounted_lights",
    "recessed_lights",
    "sconces",
    "spot_lights",
    "table_lamps"
]

f = FeatureExtractor(
    model=ResNetBottleneckModel,
    imageset_dir=media_dir,
    table_name="smart_pot",
    types=types,
    load_names=False
)

pca_path = os.path.join(os.path.dirname(__file__), "..", "smart_pot", "utils",
                    "pca_{0}_{1}_to_{2}.pickle".format("ResNet_bottleneck_model", 2048, 512))

with open(pca_path, "rb") as file:
    pca = pickle.load(file)

f.create_training_dataset(pca=pca)

# m = AutoencoderModel()
# m.load_dataset(dir_path=media_dir, db=f.db, table="smart_pot")
