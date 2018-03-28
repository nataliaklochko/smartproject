from keras import optimizers
from training.triplet_net import triplet_loss, build_model, train
from feature_extractor import FeatureExtractor
from model.resnet_bottleneck import ResNetBottleneckModel


types = [
    "free_standing_lights",
    "chandeliers",
    "mounted_lights",
    "recessed_lights",
    "sconces",
    "spot_lights",
    "table_lamps"
]

f = FeatureExtractor(model=ResNetBottleneckModel, imageset_dir="image_dataset", types=types)


def create_training_dataset():
    f.create_training_dataset()


def train_triplet_net():
    print("Build and compile model")
    num_epochs = 100
    num_to_freeze = 15

    optimizer = optimizers.SGD()

    model = build_model(num_to_freeze=num_to_freeze)
    model.compile(loss=triplet_loss, optimizer=optimizer)
    print(model.summary())

    model_json = model.to_json()
    with open("triplet_model/model_{}_freezed.json".format(num_to_freeze), "w") as json_file:
        json_file.write(model_json)

    model = train(model, generator=f.triplet_batch_generator(), num_epochs=num_epochs)

    model_json = model.to_json()
    json_file = open("triplet_model/vgg_triplet_net_{0}.json".format(num_epochs), "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights("triplet_model/vgg_triplet_net_{0}.h5".format(num_epochs))

if __name__ == "__main__":
    create_training_dataset()
    train_triplet_net()
