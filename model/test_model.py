import os
from smart_pot.image_processing import ImageProcessing
from model.resnet_bottleneck import ResNetBottleneckModel
from model.densenet_bottleneck import DenseNetBottleneckModel


if __name__ == "__main__":
    img_path = os.path.join("..", "media", "catalog_file_33261_l.jpg")
    img_prep = ImageProcessing(model=DenseNetBottleneckModel, dims=[1024, 512], find_pca=False)
    features = img_prep.get_features(img_path)
    print(features)
