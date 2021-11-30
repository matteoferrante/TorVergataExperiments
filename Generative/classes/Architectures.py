import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import numpy as np


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the image in the latent space."""

    def call(self, inputs):
        z_mean, z_log_var = inputs


        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]


        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class ResidualBlock(keras.layers.Layer):
    """A residual block"""

    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.conv1 = Conv2D(filters=filters, kernel_size=3, padding="same")
        self.bn = BatchNormalization(axis=-1)
        self.activation = Activation("relu")

        self.conv2 = Conv2D(filters=filters, kernel_size=3, padding="same")
        self.bn2 = BatchNormalization(axis=-1)
        self.skip = Conv2D(filters, kernel_size=1)
        self.last_activation = LeakyReLU(alpha=0.2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"filters": self.filters})

        return config

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.skip(x)
        x = add([inputs, x])
        x = self.last_activation(x)
        return x


class ConvResBlock(keras.layers.Layer):
    """
    Convolutional block followed by residual ones to make the network deeper
    """
    def __init__(self,n_conv_filters,n_res_block,n_res_filters):
        """

        :param n_conv_filters: int, number of convolutional filters
        :param n_res_block: int, number of residual layers
        :param n_res_filters: int, number of filters for conovlutional layers inside the residual blocks
        """
        super().__init__()
        self.n_conv_filters=n_conv_filters
        self.n_res_block=n_res_block
        self.n_res_filters=n_res_filters

        self.conv = Conv2D(filters=n_conv_filters,strides=2, kernel_size=3, padding="same") #downsample
        self.res_blocks=[]
        for i in range(n_res_block):
            self.res_blocks.append(ResidualBlock(n_res_filters))

    def get_config(self):
        config = super().get_config().copy()
        config.update({"n_conv_filters": self.n_conv_filters,
                       "n_res_block": self.n_res_block,
                       "n_res_filters":self.n_res_filters})

        return config

    def call(self, inputs, *args, **kwargs):
        x=self.conv(inputs)
        for res_block in self.res_blocks:
            x=res_block(x)

        return x



class ConvTransposeResBlock(keras.layers.Layer):
    """
    Convolutional Transpose block followed by residual ones to make the network deeper
    """
    def __init__(self,n_conv_filters,n_res_block,n_res_filters):
        """

        :param n_conv_filters: int, number of convolutional filters
        :param n_res_block: int, number of residual layers
        :param n_res_filters: int, number of filters for conovlutional layers inside the residual blocks, should be the same as conv_layers
        """
        super().__init__()
        self.n_conv_filters=n_conv_filters
        self.n_res_block=n_res_block
        self.n_res_filters=n_res_filters

        self.conv = Conv2DTranspose(filters=n_conv_filters,strides=2, kernel_size=3, padding="same") #upsample
        self.res_blocks=[]
        for i in range(n_res_block):
            self.res_blocks.append(ResidualBlock(n_res_filters))

    def get_config(self):
        config = super().get_config().copy()
        config.update({"n_conv_filters": self.n_conv_filters,
                       "n_res_block": self.n_res_block,
                       "n_res_filters":self.n_res_filters})

        return config

    def call(self, inputs, *args, **kwargs):
        x=self.conv(inputs)
        for res_block in self.res_blocks:
            x=res_block(x)

        return x







def Encoder(input_shape,latent_dim,version,conv_layer_list,residual_layer_list=None):
    """
    Return an encoder network
    :param input_shape: tuple, the shape of the input
    :param latent_dim: dimension of the latent space
    :param version: string, could be "vae" of "vqvae".
                    if "vae" it outputs a sampled vector and the parameter of the gaussian
                    if "vq vae" it outputs a Conv2D layer with latent_dim filters

    :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)

    for example
    conv_layer_list= [(2,64),(3,128),(1,256)]


    will result in a three block network
    block 1: Conv (64) -> Res(50) -> Res(50)
    block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
    block 3: conv (256) -> Res(200)

    :return: encoder model
    """


    input=Input(shape=input_shape,name="image_input")

    start=True
    for conv_filters in conv_layer_list:
        n_res_block,n_filters=conv_filters
        if start:
            x=ConvResBlock(n_filters,n_res_block,n_filters)(input)            #the first variable is called input
            start=False
        else:
            x = ConvResBlock(n_filters, n_res_block, n_filters)(x)


    if version=="vae":

        # Flatten
        x = Flatten()(x)

        # Sampling
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(input, [z_mean, z_log_var, z], name="encoder")
        return encoder

    elif version=="vqvae":
        encoder_outputs = Conv2D(latent_dim, 1, padding="same")(x)
        encoder = Model(input, encoder_outputs, name="encoder")
        return encoder


    else:
        raise NotImplemented


def ConditionalEncoder(input_shape, latent_dim, conditional_shape,embedding_dim, n_classes, version, conv_layer_list, residual_layer_list=None):
    """
    Return an encoder network
    :param input_shape: tuple, the shape of the input
    :param latent_dim: dimension of the latent space
    :param version: string, could be "vae" of "vqvae".
                    if "vae" it outputs a sampled vector and the parameter of the gaussian
                    if "vq vae" it outputs a Conv2D layer with latent_dim filters

    :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)

    for example
    conv_layer_list= [(2,64),(3,128),(1,256)]


    will result in a three block network
    block 1: Conv (64) -> Res(50) -> Res(50)
    block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
    block 3: conv (256) -> Res(200)

    :return: encoder model
    """




    condition_input = Input(shape=conditional_shape,name="condition")  # input for condition the class
    con = Embedding(n_classes, embedding_dim,name="condition_embeddings")(condition_input)
    con = Dense(np.prod(input_shape))(con)
    con = Reshape((input_shape[0], input_shape[1], 1))(con)  # produce image compatible shapes

    input = Input(shape=input_shape,name="image_input")
    merge = Concatenate()([input, con])

    start = True
    for conv_filters in conv_layer_list:
        n_res_block, n_filters = conv_filters
        if start:
            x = ConvResBlock(n_filters, n_res_block, n_filters)(merge)  # the first variable is called input
            start = False
        else:
            x = ConvResBlock(n_filters, n_res_block, n_filters)(x)

    if version == "vae":

        #Flatten
        x = Flatten()(x)

        # Sampling
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model([input,condition_input], [z_mean, z_log_var, z], name="conditional_encoder")
        return encoder

    elif version == "vqvae":
        encoder_outputs = Conv2D(latent_dim, 1, padding="same",name="encoded_inputs")(x)
        encoder = Model([input,condition_input], encoder_outputs, name="conditional_encoder")
        return encoder


    else:
        raise NotImplemented




def Discriminator(input_shape,n_classes,conv_layer_list,dense=None,dropout=0.3,activation=None):
    """
    Return a discriminator network
    :param input_shape: tuple, the shape of the input
    :param n_classes: int, number of output classes
    :param dropout: float, percentage of dropout
    :param dense: int, number of units before the last dense layer
    :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
    :param activation: string, could be "relu" or "softmax" or "tanh" for example

    for example
    conv_layer_list= [(2,64),(3,128),(1,256)]


    will result in a three block network
    block 1: Conv (64) -> Res(50) -> Res(50)
    block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
    block 3: conv (256) -> Res(200)

    :return: discriminator model
    """


    input=Input(shape=input_shape,name="image_input")

    start=True
    for conv_filters in conv_layer_list:
        n_res_block,n_filters=conv_filters
        if start:
            x=ConvResBlock(n_filters,n_res_block,n_filters)(input)            #the first variable is called input
            start=False
        else:
            x = ConvResBlock(n_filters, n_res_block, n_filters)(x)


    # Flatten
    x = Flatten()(x)

    if dense is not None:
        x = Dense(dense)(x)
        x = Dropout(dropout)(x)
        x = LeakyReLU(alpha=0.2)(x)


    x = Dense(n_classes)(x)

    if activation is not None:
        x=Activation(activation)(x)

    discriminator = Model(input, x, name="discriminator")

    return discriminator


def ConditionalDiscriminator(input_shape,conditional_shape,embedding_dim, n_classes, conv_layer_list, dropout=0.3, activation=None):
    """
    Return a discriminator network
    :param input_shape: tuple, the shape of the input
    :param n_classes: int, number of output classes
    :param dropout: float, percentage of dropout
    :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
    :param activation: string, could be "relu" or "softmax" or "tanh" for example

    for example
    conv_layer_list= [(2,64),(3,128),(1,256)]


    will result in a three block network
    block 1: Conv (64) -> Res(50) -> Res(50)
    block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
    block 3: conv (256) -> Res(200)

    :return: discriminator model
    """

    input = Input(shape=input_shape, name="image_input")


    condition_input = Input(shape=conditional_shape,name="condition")  # input for condition the class
    con = Embedding(n_classes, embedding_dim,name="condition_embeddings")(condition_input)
    con = Dense(np.prod(input_shape))(con)
    con = Reshape((input_shape[0], input_shape[1], 1))(con)  # produce image compatible shapes

    merge=Concatenate()([input,con])

    start = True
    for conv_filters in conv_layer_list:
        n_res_block, n_filters = conv_filters
        if start:
            x = ConvResBlock(n_filters, n_res_block, n_filters)(merge)  # the first variable is called input
            start = False
        else:
            x = ConvResBlock(n_filters, n_res_block, n_filters)(x)

    # Flatten
    x = Flatten()(x)

    x = Dense(4096)(x)
    x = Dropout(dropout)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(n_classes)(x)

    if activation is not None:
        x = Activation(activation)(x)

    discriminator = Model([input,condition_input], x, name="conditional_discriminator")

    return discriminator




def Decoder(target_shape,latent_dim,conv_layer_list,encoder_output_shape=None,version="vae"):
    """
    Return a decoder/generator network
    :param input_shape: tuple, the shape of the input
    :param latent_dim: dimension of the latent space
    :param version: string, could be "vae" of "vqvae".
                    if "vae" it outputs a sampled vector and the parameter of the gaussian
                    if "vq vae" it outputs a Conv2D layer with latent_dim filters

    :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
    :param encoder_output_shape: tuple, required if version is "vqvae"
    for example
    conv_layer_list= [(2,64),(3,128),(1,256)]


    will result in a three block network
    block 1: Conv (64) -> Res(50) -> Res(50)
    block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
    block 3: conv (256) -> Res(200)

    :return: decoder model
    """

    # infer the starting dimension.
    target_shape_side=target_shape[0]
    startDim=target_shape_side//2**len(conv_layer_list)

    start = True



    if version=="vqvae":
        latent_inputs = keras.Input(shape=encoder_output_shape,name="latent_input")
    else:
        latent_inputs = keras.Input(shape=(latent_dim,),name="latent_input")
        x = Dense(startDim * startDim * 64)(latent_inputs)
        x = Reshape((startDim, startDim, 64))(x)
        x = LeakyReLU()(x)
        start=False

    for conv_filters in conv_layer_list:
        n_res_block, n_filters = conv_filters
        if start:
            x = ConvTransposeResBlock(n_filters, n_res_block, n_filters)(latent_inputs)  # the first variable is called input
            start = False
        else:
            x = ConvTransposeResBlock(n_filters, n_res_block, n_filters)(x)

    decoder_outputs = Conv2DTranspose(target_shape[-1], 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder



def ConditionalDecoder(target_shape,latent_dim,conditional_shape,embedding_dim, n_classes,conv_layer_list,encoder_output_shape=None,version="vae"):
    """
    Return a decoder/generator network
    :param input_shape: tuple, the shape of the input
    :param latent_dim: dimension of the latent space
    :param version: string, could be "vae" of "vqvae".
                    if "vae" it outputs a sampled vector and the parameter of the gaussian
                    if "vq vae" it outputs a Conv2D layer with latent_dim filters

    :param conv_layer_list: list, each element is a tuple in the form (n_res_block, n_res_filters)
    :param encoder_output_shape: tuple, required if version is "vqvae"
    for example
    conv_layer_list= [(2,64),(3,128),(1,256)]


    will result in a three block network
    block 1: Conv (64) -> Res(50) -> Res(50)
    block 2: Conv (128) -> Res(100) -> Res(100) -> Res(100)
    block 3: conv (256) -> Res(200)

    :return: decoder model
    """

    # infer the starting dimension.
    target_shape_side=target_shape[0]
    startDim=target_shape_side//2**len(conv_layer_list)



    if version=="vqvae":
        latent_inputs = keras.Input(shape=encoder_output_shape,name="latent_input")
        in_label = Input(shape=conditional_shape)  # input for condition the class
        li = Embedding(n_classes, embedding_dim)(in_label)  # li stands for label input
        li = Dense(encoder_output_shape[0]*encoder_output_shape[1])(li)
        li = Reshape((encoder_output_shape[0], encoder_output_shape[1], 1))(li)  # produce image compatible shapes

        x = Concatenate()([latent_inputs, li])  # concatenate into a multichannel image


    else:
        latent_inputs = keras.Input(shape=(latent_dim,),name="latent_input")
        x = Dense(startDim * startDim * 64)(latent_inputs)
        x = Reshape((startDim, startDim, 64))(x)
        x = LeakyReLU()(x)

        in_label = Input(shape=conditional_shape)  # input for condition the class
        li = Embedding(n_classes, embedding_dim)(in_label)  # li stands for label input
        li = Dense(np.prod(startDim*startDim))(li)
        li = Reshape((startDim, startDim, 1))(li)  # produce image compatible shapes

        x = Concatenate()([x, li])  # concatenate into a multichannel image



    for conv_filters in conv_layer_list:
        n_res_block, n_filters = conv_filters

        x = ConvTransposeResBlock(n_filters, n_res_block, n_filters)(x)

    decoder_outputs = Conv2DTranspose(target_shape[-1], 3, activation="sigmoid", padding="same")(x)

    decoder = Model([latent_inputs,in_label], decoder_outputs, name="conditional_decoder")
    return decoder

def VGGNetLike(input_shape=(128,128,3),n_attributes=40):

    input=Input(shape=input_shape)

    #block 1
    x=Conv2D(64,3,padding="same")(input)
    x=Activation("relu")(x)

    x=Conv2D(64,3,padding="same")(x)
    x=Activation("relu")(x)



    #block 2
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(128, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(128, 3, padding="same")(x)
    x = Activation("relu")(x)

    #block 3
    x = MaxPooling2D()(x)

    x = Conv2D(256, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(256, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(256, 3, padding="same")(x)
    x = Activation("relu")(x)

    #block 4

    x = MaxPooling2D()(x)

    x = Conv2D(512, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(512, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(512, 3, padding="same")(x)
    x = Activation("relu")(x)

    #block 5

    x = MaxPooling2D()(x)

    x = Conv2D(512, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(512, 3, padding="same")(x)
    x = Activation("relu")(x)

    x = Conv2D(512, 3, padding="same")(x)
    x = Activation("relu")(x)

    x=Flatten()(x)

    x=Dense(4096)(x)
    x=Activation("relu")(x)
    x=Dense(n_attributes)(x)

    model=Model(input,x,name="VGGNetLike_Classificator")
    return model

