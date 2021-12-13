
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

import numpy as np

from Generative.classes.Architectures import ConditionalEncoder


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the image in the latent space."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon




class CVAE(keras.Model):

    """Class for Conditional Variational Autoencoder extending keras.model

    more in-depth explaination could be found at: https://agustinus.kristia.de/techblog/2016/12/17/conditional-vae/


    Theory:
    --------

    Same in VAE but we want to condition the distributions on c

    encoder=Q(z|X,c)
    decoder=P(X|z,c) so basically we have one p(z) for each possible conditions


    """



    def __init__(self, input_dim, latent_dim,n_classes=10,emb_dim=50,output_channel=1,encoder_architecture=[(0,128),[(0,256)]], decoder_architecture=[(0,128),[(0,256)]], conditional_shape=(1,), **kwargs):
        """

        :param input_dim: dimension of images
        :param latent_dim: latent_dim

        Attributes
        ----------
        total_loss_tracker: mean of the sum of reconstruction_loss and kl_loss
        reconstruction_loss: mean metrics that are L2 norm between input and outputs
        kl_loss: regularizer loss



        """
        super(CVAE, self).__init__(**kwargs)

        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.n_classes=n_classes
        self.emb_dim=emb_dim

        self.condition_shape=conditional_shape


        self.encoder_architecture=encoder_architecture
        self.decoder_architecture=decoder_architecture

        self.encoder=ConditionalEncoder(self.input_dim,latent_dim=latent_dim,conditional_shape=conditional_shape,n_classes=n_classes,embedding_dim=emb_dim,version="vae",conv_layer_list=encoder_architecture)
        self.decoder=ConditionalEncoder(self.input_dim,latent_dim=latent_dim,conditional_shape=conditional_shape,embedding_dim=emb_dim,n_classes=n_classes,version="vae",conv_layer_list=decoder_architecture)


#        self.encoder = self.build_encoder(input_dim,latent_dim)
 #       self.decoder = self.build_decoder(latent_dim,output_channel=1)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):

        """
        overload of train_step of keras.fit method

        # 1. encoder the images into the latent space and sample a vector.

        # 2. reconstruct the images.

        # 3. compute the recostruction loss, for example using L2 norm or binary_crossentropy between data and reconstruction

        # 4. compute the kl_divergence. If we assume a normal prior the kl term can be espressed as -0.5*(1+ z_log_var - z_mean^2 -e^z_log_var)

        :param data: images to be reconstructed
        :return: metrics
        """


        with tf.GradientTape() as tape:

            img,conditions=data
            z_mean, z_log_var, z = self.encoder([img,conditions])

            reconstruction = self.decoder([z,conditions])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(img, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        img, conditions = data

        z_mean, z_log_var, z = self.encoder([img,conditions])
        reconstruction = self.decoder([z,conditions])
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(img, reconstruction)
        )
        reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, data):
        x,cond=data
        z_mean, z_log_var, z = self.encoder([x,cond])
        reconstruction = self.decoder([z,cond])
        return reconstruction

    def decode(self,z,conditions):
        return self.decoder([z,conditions])

    def build_encoder(self,input_dim, latent_dim,n_classes=10,emb_dim=50, chanDim=-1):


        ## label input

        condition_input=Input(shape=(1,))                               #input for condition the class
        con=Embedding(n_classes,emb_dim)(condition_input)
        con=Dense(np.prod(input_dim))(con)
        con = Reshape((input_dim[0], input_dim[1], 1))(con)                 #produce image compatible shapes




        encoder_inputs = keras.Input(shape=input_dim)

        merge = Concatenate()([encoder_inputs, con])
        # first block
        x = Conv2D(32, (3, 3), padding="same")(merge)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # second block
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # third block
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Flatten
        x = Flatten()(x)

        # Sampling
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model([encoder_inputs,condition_input], [z_mean, z_log_var, z], name="encoder")
        # encoder.summary()
        return encoder


    def build_decoder(self,latent_dim,n_classes=10,emb_dim=50, chanDim=-1, startDim=7,output_channel=1):


        ## label input

        condition_input=Input(shape=(1,))                              #input for condition the class
        con=Embedding(n_classes,emb_dim)(condition_input)               #li stands for label input
        con=Dense(startDim * startDim )(con)
        con = Reshape((startDim, startDim, 1))(con)       #produce image compatible shapes



        latent_inputs = keras.Input(shape=(latent_dim,))



        x = Dense(startDim * startDim * 64)(latent_inputs)
        x = Reshape((startDim, startDim, 64))(x)

        merge = Concatenate()([x, con])                                #maybe i can concatenate the data before


        x = LeakyReLU()(merge)


        ## to work with mnist

        #x = Conv2DTranspose(128, 3, strides=2, padding="same")(x)
        #x = BatchNormalization(axis=chanDim)(x)
        #x = LeakyReLU()(x)

        x = Conv2DTranspose(128, 3, strides=2, padding="same")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(64, 3, strides=2, padding="same")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU()(x)

        decoder_outputs = Conv2DTranspose(output_channel, 3, activation="sigmoid", padding="same")(x)
        decoder = Model([latent_inputs,condition_input], decoder_outputs, name="decoder")

        return decoder

