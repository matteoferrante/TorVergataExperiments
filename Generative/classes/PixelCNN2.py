"""The exact same file as PixelCNN but for VQ VAE 2 including top and bottom hierchical embeddings"""



import json

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import *

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


class ConditionalPixelCNN2(keras.Model):

    """Class that extends keras model to implement PixelCNN autoregressive model
    reference paper: https://arxiv.org/pdf/1606.05328.pdf

    """

    def __init__(self,input_top_dim=(4,4),input_bottom_dim=(8,8),num_embeddings=128,n_residual=5,n_convlayer=2,ksize=7,n_classes=10,cond_emb=50):
        """

        :param input_dim: input dimension (should be the same as the embedding space
        :param n_embeddings: number of embeddings
        :param n_residual: number of residual layers
        :param n_convlayer: number of PixelConvLayer
        :param ksize: kernel size of the input Layer
        :param sampler: Model defined once pixelCnn is trained with build_sampler method

        :param n_classes: int, number of possible conditions
        :cond_emb: int, dimension of the learned embedding space for conditions
        """
        super(ConditionalPixelCNN2, self).__init__()
        self.input_top_dim=input_top_dim
        self.input_bottom_dim = input_bottom_dim

        self.num_embeddings=num_embeddings
        self.n_residual=n_residual
        self.n_convlayer=n_convlayer
        self.ksize=ksize

        self.n_classes=n_classes
        self.cond_emb=cond_emb

        self.model=self.build_pixelcnn()
        #self.model.build()
        #print(self.model.summary())


    def build_pixelcnn(self):
        """build the conditional model itself"""

        top_inputs = keras.Input(shape=self.input_top_dim, dtype=tf.int32)
        top_ohe = tf.one_hot(top_inputs, self.num_embeddings)
        condition_input_top=Input(shape=(1,))                                             #input for condition the class


        con=Embedding(self.n_classes,self.cond_emb)(condition_input_top)
        con=Dense(np.prod(self.input_top_dim))(con)
        con = Reshape((self.input_top_dim[0], self.input_top_dim[1], 1))(con)                     #produce image compatible shapes




        merge_top = Concatenate(axis=-1)([top_ohe, con])                                           #until now we have conditioned the top input with the label


        bottom_inputs=Input(shape=self.input_bottom_dim,dtype=tf.int32)
        bottom_ohe = tf.one_hot(bottom_inputs, self.num_embeddings)

        enlarged_top_condition=Conv2DTranspose(1,strides=2,padding="same",kernel_size=3)(top_ohe)

        merge_bottom=Concatenate()([bottom_ohe,enlarged_top_condition])

        ## TOP PIXELCNN

        x_t=PixelConvLayer(mask_type="A",filters=128,kernel_size=self.ksize,activation="relu",padding="same")(merge_top)

        #add residual blocks
        for _ in range(self.n_residual):
            x_t=ResidualBlock(filters=128)(x_t)

        #add pixelcnn
        for _ in range(self.n_convlayer):
            x_t=PixelConvLayer(mask_type="B",filters=128,kernel_size=1,strides=1,activation="relu",padding="valid")(x_t)


        ##BOTTOM PIXEL CNN

        x_b = PixelConvLayer(mask_type="A", filters=128, kernel_size=self.ksize, activation="relu", padding="same")(merge_bottom)

        # add residual blocks
        for _ in range(self.n_residual):
            x_b = ResidualBlock(filters=128)(x_b)

        # add pixelcnn
        for _ in range(self.n_convlayer):
            x_b = PixelConvLayer(mask_type="B", filters=128, kernel_size=1, strides=1, activation="relu",padding="valid")(x_b)

        #x=Concatenate()([x_t,x_b])
        out_top=Conv2D(self.num_embeddings,kernel_size=1,strides=1,padding="valid")(x_t)
        out_bottom=Conv2D(self.num_embeddings,kernel_size=1,strides=1,padding="valid")(x_b)

        pixel_cnn=Model([top_inputs,bottom_inputs,condition_input_top],[out_top,out_bottom])
        return pixel_cnn


    def call(self,x, *args, **kwargs):
        """
        :param x: x should be a tuple (codebook, condition)
        :return: output of the model
        """
        return self.model(x)

    def save_dict(self,path):
        dictionary_data={"input_top_dim":self.input_top_dim,"input_bottom_dim":self.input_bottom_dim,"num_embeddings":self.num_embeddings,
                         "n_residual":self.n_residual,"n_convlayer":self.n_convlayer,
                         "ksize":self.ksize, "n_classes":self.n_classes,
                         "cond_emb":self.cond_emb
                         }
        a_file = open(path, "w")
        json.dump(dictionary_data, a_file)
        a_file.close()