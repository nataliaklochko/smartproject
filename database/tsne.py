import numpy as np
import pickle

from sklearn.manifold import TSNE

from database.data_base import DataBase
from model.resnet_bottleneck import ResNetBottleneckModel
from model.nasnet_bottleneck import NASNetBottleneckModel
from model.vgg_bottleneck import VGGBottleneckModel
from model.vgg_bottleneck_fine_tuned import VGGBottleneckModelFineTuned
from model.autoencoder import AutoencoderModel


db = DataBase()
model = AutoencoderModel()

db.c.execute("SELECT ID, name, {0}  FROM smart_pot".format(model.name))
data = db.c.fetchall()

vectors = []
indices = []
names = []

for item in data:
    vectors.append(np.frombuffer(item[2], dtype=np.float32))
    indices.append(item[0] - 1)
    names.append(item[1])

# with open("../smart_pot/utils/pca_ResNet_bottleneck_model_2048_to_512.pickle", "rb") as file:
#     pca = pickle.load(file)
# vectors = pca.transform(vectors)

tsne = TSNE(n_components=2, perplexity=50, verbose=1)
tsne.fit(vectors)

with open("utils/tsne_2d_{}.pickle".format(model.name), "wb") as file:
    pickle.dump(tsne, file)
