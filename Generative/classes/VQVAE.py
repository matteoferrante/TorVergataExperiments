import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import numpy as np

class VectorQuantizer(Layer):

    """This is the core object of the VQ-VAE Architecture


    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        """

        :param num_embeddings: number of vectors in the codebook
        :param embedding_dim:  dimension of the latent vectors
        :param beta:           weight parameter that controls how flexible are the values in the codebook. This should be in 0.25,2
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):


        """

        Explanation:
        -------------

        Commitment_loss: this is  |sg(z)-e| term. The gradient are stopped at encoded vectors so we try to make codebook vectors similar to encoded ones
        codebook_loss: this is  beta*|z-sg(e)| term. We freeze the codebook vectors and we try to push the encoded vectors to the codebook ones

        If beta is greater than we want to keep our codebook vectors more rigid.

        """

        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)



        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)         #trick to copy-past gradients
        return quantized

    def get_code_indices(self, flattened_inputs):

        """Try to find the closest vector in the embedding to our encoded vector. It's argmin of ||z-e_j||^2"""

        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
                tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                + tf.reduce_sum(self.embeddings ** 2, axis=0)
                - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class VQVAE(keras.Model):
    """Class to implement vector quantized variational autoencoder
     from article: https://arxiv.org/abs/1711.00937

    The idea is to use a discrete space as latent space. This is called the codebook and it's learnable.
    This space consist in num_embeddings of latent_dim (for example: num_embeddings=100 with latent dim 32 means
    that the codebook is composed by 100 unique vectors with 32 components).

    The prior is learnable instead of a static gaussian (like p(z) in VAEs) and the latent space in quantized)

    loss function is basiacally the sum of three terms:

    reconstruction_loss+commitment_loss+codebook_loss.

    Decoder optimizes the first term,
    Encoder the first and the last term
    Embedding are optimized by the middle loss term

     """

    def __init__(self, input_dim, latent_dim=32, num_embeddings=128,train_variance=1, **kwargs):
        """

        :param input_dim: input image dimension
        :param latent_dim: latent dimension in the embedding space
        :param num_embeddings: number of discrete vectors in the codebook
        :param train_variance: hyper-parameters to weight the reconstruction loss
        """

        super(VQVAE, self).__init__(**kwargs)
        self.input_dim=input_dim
        self.latent_dim = latent_dim
        self.train_variance = train_variance

        self.num_embeddings = num_embeddings

        self.encoder=self.get_encoder()
        self.decoder=self.get_decoder()

        self.vqvae = self.get_vqvae()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]


    def __call__(self, data):
        return self.vqvae(data)

    def train_step(self, x):
        """

        :param x: images
        :return: metrics
        """
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)            #call the vq vae, which records also the commitment and the codebook losses

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    def test_step(self, data):

        #TODO: sostituire compiled_loss con il calcolo esplicito!

        # Unpack the data
        x=data
        # Compute predictions
        y_pred = self.vqvae(x)

        reconstruction_loss = (
                tf.reduce_mean((x - y_pred) ** 2) / self.train_variance
        )



        return {"val_reconstruction_loss": reconstruction_loss}

    def get_encoder(self):
        encoder_inputs = keras.Input(shape=self.input_dim)
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(
            encoder_inputs
        )
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        encoder_outputs = Conv2D(self.latent_dim, 1, padding="same")(x)

        model=Model(encoder_inputs, encoder_outputs, name="encoder")
        return model

    def get_decoder(self):
        latent_inputs = keras.Input(shape=self.get_encoder().output.shape[1:])



        ## to work with mnist

        #x = Conv2DTranspose(128, 3, strides=2, padding="same")(x)
        #x = BatchNormalization(axis=chanDim)(x)
        #x = LeakyReLU()(x)

        x = Conv2DTranspose(128, 3, strides=2, padding="same")(latent_inputs)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(64, 3, strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        return decoder


    def get_vqvae(self):
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.latent_dim, name="vector_quantizer")
        encoder = self.encoder
        decoder = self.decoder
        #inputs = Input(shape=self.input_dim)
        inputs=self.encoder.inputs
        encoder_outputs = encoder(inputs)
        quantized_latents = self.vq_layer(encoder_outputs)
        reconstructions = decoder(quantized_latents)

        model=keras.Model(inputs, reconstructions, name="vq_vae")
        return model


    def load_weights(self,encoder_w,embeddings_w,decoder_w):
        """

        :param encoder_w: encoder weights
        :param embeddings_w: embeddings
        :param decoder_w: decoder weights
        :return:
        """
        self.encoder.load_weights(encoder_w)
        self.vq_layer.set_weights(np.load(embeddings_w))
        self.decoder.load_weights(decoder_w)

