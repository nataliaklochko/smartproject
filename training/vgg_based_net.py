import os

from keras.applications.vgg16 import VGG16

from keras.engine.topology import Input
from keras import metrics, Model
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import SGD, Adam

from keras import backend as K

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator


DATASET_DIR = "image_dataset_balanced"


def build_model(model_weights="vgg_model/model_0_weights-improvement-396-0.27.hdf5", num_to_freeze=17):

    with K.name_scope("VGG16"):
        input_shape = (224, 224, 3)
        base_input = Input(shape=input_shape)

        base_model = VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=base_input,
            pooling="avg"
        )

    for layer in base_model.layers[:num_to_freeze]:
        layer.trainable = False

    with K.name_scope("Classifier"):
        x = base_model.output
        x = Dense(256, activation="relu")(x)
        # x = Dropout(0.1)(x)
        x = Dense(128, activation="relu")(x)
        predictions = Dense(7, activation="softmax")(x)
        model = Model(input=base_model.input, output=predictions)

    for layer in model.layers[-3:]:
        print(layer)
        layer.trainable = False

    if model_weights:
        model.load_weights(model_weights)

    # optimizer = SGD(lr=0.001, momentum=0.001)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[metrics.categorical_accuracy]
    )

    model_json = model.to_json()
    with open(os.path.join("vgg_model", "model_0.json"), "w") as json_file:
        json_file.write(model_json)

    print(model.summary())

    return model


def train(model, batch_size, num_epochs=100):

    train_data_dir = os.path.join(os.path.dirname(__file__), DATASET_DIR, "train")
    validation_data_dir = os.path.join(os.path.dirname(__file__), DATASET_DIR, "test")
    target_size = (224, 224)

    # nb_train_samples = 100600
    # nb_validation_samples = 11164

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    filepath="model_0_weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"

    checkpointer = ModelCheckpoint(
        filepath=os.path.join("vgg_model", filepath),
        monitor="loss",
        verbose=1
    )

    tensorboard = TensorBoard(
        log_dir='./vgg_logs',
        write_images=True,
        batch_size=batch_size
    )

    model.fit_generator(
        generator=train_generator,
        # steps_per_epoch=nb_train_samples // batch_size,
        steps_per_epoch=128,
        epochs=num_epochs,
        validation_data=validation_generator,
        # validation_steps=nb_validation_samples // batch_size,
        validation_steps=128,
        callbacks=[tensorboard],
        # initial_epoch=397
    )

    return model


if __name__ == "__main__":
    model = build_model()
    train(model, batch_size=32, num_epochs=1000)
