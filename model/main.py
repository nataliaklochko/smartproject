import os

from model.vgg_bottleneck import VGGBottleneckModel
from model.resnet_bottleneck import ResNetBottleneckModel
from model.densenet_bottleneck import DenseNetBottleneckModel
from model.nasnet_bottleneck import NASNetBottleneckModel
from model.vgg_bottleneck_fine_tuned import VGGBottleneckModelFineTuned
from model.autoencoder import AutoencoderModel

from feature_extractor import FeatureExtractor


media_dir = os.path.join("..", "media")
f = FeatureExtractor(
    model=AutoencoderModel,
    imageset_dir=media_dir,
    table_name="smart_pot",
    load_names=False
)

m = VGGBottleneckModel()
m.load_dataset(dir_path=media_dir, db=f.db, table="smart_pot")
