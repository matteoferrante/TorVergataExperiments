import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import *

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

##TODO: fix PixelCNN sampling and dimensions when loeaded


class TfDistPixelCNN:


    """This is a class for autoregressive model like pixelCNN
    PixelCNN is an autoregressive model that paramtetrize the predicted value of a pixel i with the previous value 0,1,i-1
    This implementation is based on https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PixelCNN

    As alternative is possible to follow: https://keras.io/examples/generative/pixelcnn/ for an implementation with keras
    """

    def __init__(self,input_dim=(7,7),num_resnet=2,num_hierarchies=1,num_filters=128,num_logistic_mix=3,dropout_p=.3):
        self.input_dim=input_dim
        self.num_resnet=num_resnet
        self.num_hierarchies=num_hierarchies
        self.num_filters=num_filters
        self.num_logistic_mix=num_logistic_mix
        self.dropout_p=dropout_p
        self.model=self.build_model()

    def build_model(self):
        self.dist=tfp.distributions.PixelCNN(image_shape=(self.input_dim[0],self.input_dim[1],1),num_resnet=self.num_resnet,num_hierarchies=self.num_hierarchies,num_filters=self.num_filters,num_logistic_mix=self.num_logistic_mix,dropout_p=self.dropout_p)

        self.input=tfkl.Input(self.input_dim)
        self.log_prob=self.dist.log_prob(self.input)


        model=Model(inputs=self.input,outputs=self.log_prob)
        model.add_loss(-tf.reduce_mean(self.log_prob))
        return model

    def compile(self,optimizer=keras.optimizers.Adam(1e-3),metrics=[]):
        self.model.compile(optimizer=optimizer,metrics=metrics)

    def fit(self,data,epochs=10,batch_size=128,validation_split=0.1,callbacks=[]):
        self.model.fit(data,data,batch_size=batch_size,epochs=epochs,validation_split=validation_split,callbacks=callbacks)

    def sample(self,n_images):
        return self.dist.sample(n_images)


class PixelConvLayer(keras.layers.Layer):
    """
    This is just a Convolutional layer but includes masking.
    If we are trying to predict the i-th element of a sequence 0,1..i-1 we will use a mask [1,1,..1,0,0..0]
    """
    def __init__(self, mask_type, filters=128,kernel_size=7,activation="relu",**kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.filters=filters
        self.kernel_size=kernel_size=kernel_size
        self.activation=activation
        self.conv = Conv2D(filters=filters,kernel_size=kernel_size,activation=activation,**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mask_type': self.mask_type,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation
        })
        return config

class ResidualBlock(keras.layers.Layer):
    """A residual block based on PixelConvLayers"""
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters=filters
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])

    def get_config(self):

        config = super().get_config().copy()
        config.update({"filters":self.filters})
        return config


class PixelCNN(keras.Model):

    """Class that extends keras model to implement PixelCNN autoregressive model
    reference paper: https://arxiv.org/pdf/1606.05328.pdf

    """

    def __init__(self,input_dim=(7,7),n_embeddings=128,n_residual=5,n_convlayer=2,ksize=7):
        """

        :param input_dim: input dimension (should be the same as the embedding space
        :param n_embeddings: number of embeddings
        :param n_residual: number of residual layers
        :param n_convlayer: number of PixelConvLayer
        :param ksize: kernel size of the input Layer
        :param sampler: Model defined once pixelCnn is trained with build_sampler method
        """
        super(PixelCNN, self).__init__()
        self.input_dim=input_dim
        self.num_embeddings=n_embeddings
        self.n_residual=n_residual
        self.n_convlayer=n_convlayer
        self.ksize=ksize

        self.model=self.build_pixelcnn()
        self.model.build(input_shape=input_dim)
        print(self.model.summary())

    def build_pixelcnn(self):
        """build the model itself"""

        inputs = keras.Input(shape=self.input_dim, dtype=tf.int32)
        ohe = tf.one_hot(inputs, self.num_embeddings)

        x=PixelConvLayer(mask_type="A",filters=128,kernel_size=self.ksize,activation="relu",padding="same")(ohe)

        #add residual blocks
        for _ in range(self.n_residual):
            x=ResidualBlock(filters=128)(x)

        #add pixelcnn
        for _ in range(self.n_convlayer):
            x=PixelConvLayer(mask_type="B",filters=128,kernel_size=1,strides=1,activation="relu",padding="valid")(x)

        out=Conv2D(self.num_embeddings,kernel_size=1,strides=1,padding="valid")(x)

        pixel_cnn=Model(inputs,out)
        return pixel_cnn


    def call(self,x, *args, **kwargs):
        return self.model(x)


    def get_config(self):
        return self.model.get_config()

    def from_config(self,config):
        self.model.from_config(config,custom_objects={"PixelConvLayer":PixelConvLayer,"ResidualBlock":ResidualBlock})

    def from_json(self,json):
        self.model=model_from_json(json,custom_objects={"PixelConvLayer":PixelConvLayer,"ResidualBlock":ResidualBlock})

    def load_model(self,model_path):
        self.model=load_model(model_path,custom_objects={"PixelConvLayer":PixelConvLayer,"ResidualBlock":ResidualBlock})






