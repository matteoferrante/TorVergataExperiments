
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

from tensorflow.keras.models import Model





class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the image in the latent space."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon




class VAE(keras.Model):

    """Class for Variational Autoencoder extending keras.model

    more in-depth explaination could be found at : https://arxiv.org/abs/1606.05908
    or : https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

    Theory:
    --------

    The idea of a VAE is that x=d(e(x)) where e is an encoder and d a decoder.
    The encoder map the input x in a latent space distribution p(z|x) and we want that similar inputs are close in this latent space


    We want p(z|x) gaussian and we can use bayes theorem

    p(z|x)= p(x|z) p(z)/ p(x)

    p(x|z) is a gaussian N(f(z),cI).
    Let's assume a N(0,1) for the prior p(z) so ideally we could compute p(z|x).
    Unfortunately p(x) is a sort for normalizazion that could be expressed p(x) = integrate p(x|u)p(u)du over all possible u.
    This is usally untractable so we need a function that approximate p(z|x) because we can't compute it directly.

    In Variation Inference (VI) usually we look for best approximation of a target distribution from parametrized distributions in
    a family like gaussians. We try to minimize a measure of distance between target and approximating distributions.

    So we can approximate p(z|x) with a q_x(z) which is a Gaussian N(g(x),h(x))

    We can search in the space of g and h their best values, the values that minimize the KL divergence

    g*,h*=argmin KL(q_x(z),p(z|x))

    using KL definition and bayes theorem

    g*,h*=argmin (E[log(q_x(z))] - E[log(p(x|z)] - E[log(p(z))] + E[log(p(x))])

    rearranging the terms and discarding E[log(p(z))] which is a constant

    g*,h*=argmin( E(log(p(x|z)) - KL (q_x(z),p(x))).

    E(log(p(x|z)) is just (1/2c) *||x-f(z)||^2 becuase p(x|z) is a guassian and when we found the best values to compute q
    we an use them to approximate p(x) which was the problematic quantity.




    """



    def __init__(self, input_dim, latent_dim,output_channel=1, **kwargs):
        """

        :param input_dim: dimension of images
        :param latent_dim: latent_dim

        Attributes
        ----------
        total_loss_tracker: mean of the sum of reconstruction_loss and kl_loss
        reconstruction_loss: mean metrics that are L2 norm between input and outputs
        kl_loss: regularizer loss



        """
        super(VAE, self).__init__(**kwargs)

        self.input_dim=input_dim
        self.latent_dim=latent_dim

        self.encoder = self.build_encoder(input_dim,latent_dim)
        self.decoder = self.build_decoder(latent_dim,output_channel=1)
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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
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
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, reconstruction)
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

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def decode(self,z):
        return self.decoder(z)

    def build_encoder(self,input_dim, latent_dim, chanDim=-1):
        encoder_inputs = keras.Input(shape=input_dim)

        # first block
        x = Conv2D(32, (3, 3), padding="same")(encoder_inputs)
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
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # encoder.summary()
        return encoder


    def build_decoder(self,latent_dim, chanDim=-1, startDim=7,output_channel=1):
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = Dense(startDim * startDim * 64)(latent_inputs)
        x = Reshape((startDim, startDim, 64))(x)
        x = LeakyReLU()(x)


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
        decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        return decoder

