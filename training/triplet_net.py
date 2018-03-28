import os
from keras import backend as K

from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.layers import Lambda
from keras.models import Model
from keras import callbacks



def build_model(num_to_freeze):
    """
    Builds model based on pretrained VGG-16 (top is not included)

    @param: num_to_freeze - number of VGG-16 layers which won't be fine-tuned
    @return: model - Keras model

    """

    input_shape = (224, 224, 3)
    vgg_input = Input(shape=input_shape)

    vgg_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=vgg_input,
        pooling="avg",
        input_shape=(224, 224, 3)
    )

    base_model = Model(
        outputs=vgg_model.layers[-1].output,
        inputs=vgg_model.input,
        name="lightning_vgg"
    )

    for layer in base_model.layers[:num_to_freeze]:
        print("{0} is freezed".format(layer))
        layer.trainable = False

    input_anchor = Input(shape=input_shape, name="input_anchor")
    input_positive = Input(shape=input_shape, name="input_pos")
    input_negative = Input(shape=input_shape, name="input_neg")

    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    positive_dist = Lambda(pearson_correlation, name="pos_dist")([net_anchor, net_positive])
    negative_dist = Lambda(pearson_correlation, name="neg_dist")([net_anchor, net_negative])

    stacked_dists = Lambda(
                lambda vects: K.stack(vects, axis=1),
                name="stacked_dists"
    )([positive_dist, negative_dist])

    model = Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=stacked_dists,
        name="triplet_net"
    )
    return model


def triplet_loss(_, y_pred):
    """
    Loss to compare anchor, pos and neg images

    @param: y_pred - distances anchor-pos and anchor-neg
    @return: loss for training

    """
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), - K.square(y_pred[:, 0, 0]) + K.square(y_pred[:, 1, 0]) + margin))
    # return K.mean(y_pred[:, 0, 0])


def euclidean_distance(vects):
    """
    Calculates Euclidean distance between vectors in vects

    """
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def pearson_correlation(vects):
    x, y = vects
    mean_x = K.mean(x, axis=1, keepdims=True)
    mean_y = K.mean(y, axis=1, keepdims=True)
    std_x = K.std(x, axis=1, keepdims=True)
    std_y = K.std(y, axis=1, keepdims=True)
    return K.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)


def accuracy(_, y_pred):
    return K.mean(y_pred[0] < y_pred[1])


def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)


def train(model, generator, num_epochs=200):

    filepath="weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    checkpointer = callbacks.ModelCheckpoint(
        filepath=os.path.join("triplet_model", filepath),
        monitor="loss",
        verbose=1
    )
    tensorboard = callbacks.TensorBoard(
        log_dir='./triplet_logs',
        histogram_freq=0,
        write_graph=True,
        write_images=False
    )

    model.fit_generator(
        generator=generator,
        steps_per_epoch=256,
        epochs=num_epochs,
        callbacks=[checkpointer, tensorboard]
    )
    return model
