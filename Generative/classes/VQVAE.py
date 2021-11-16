import json

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import numpy as np


class ResidualBlock(keras.layers.Layer):
    """A residual block based """
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters=filters
        self.conv1 = Conv2D(filters=filters, kernel_size=3, padding="same")
        self.bn=BatchNormalization(axis=-1)
        self.activation=Activation("relu")

        self.conv2 = Conv2D(filters=filters, kernel_size=3, padding="same")
        self.bn2 = BatchNormalization(axis=-1)
        self.skip= Conv2D(filters,kernel_size=1)
        self.last_activation=Activation("relu")

    def get_config(self):

        config = super().get_config().copy()
        config.update({"filters":self.filters})

        return config

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x=  self.activation(x)
        x=  self.conv2(x)
        x=  self.bn2(x)
        x=  self.skip(x)
        x=  add([inputs,x])
        x = self.last_activation(x)
        return x






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

    def get_config(self):

        config = super().get_config().copy()
        config.update({"embedding_dim":self.embedding_dim,"num_embeddings":self.num_embeddings,"beta":self.beta})

        return config


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




class VQVAE2(keras.Model):
    """Implementation of the hiearchical model based on https://proceedings.neurips.cc/paper/2019/file/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Paper.pdf paper
    """

    def __init__(self, input_dim, latent_dim=32, num_embeddings=128,train_variance=1,channels=64,n_res_block=2,n_res_channel=32, outchannels=3):
        """

        :param input_dim: input image dimension
        :param latent_dim: latent dimension in the embedding space
        :param num_embeddings: number of discrete vectors in the codebook
        :param train_variance: hyper-parameters to weight the reconstruction loss
        """

        super(VQVAE2, self).__init__()
        self.input_dim=input_dim
        self.latent_dim = latent_dim
        self.train_variance = train_variance
        self.channels=channels
        self.n_res_block=n_res_block
        self.n_res_channel=n_res_channel

        self.num_embeddings = num_embeddings

        self.outchannels=outchannels

        self.encoder_b=self.build_encoder(self.input_dim,channels=channels,n_res_block=n_res_block,n_res_channel=n_res_channel,level="bottom")

        self.encoder_t=self.build_encoder(self.input_dim,channels=channels,n_res_block=n_res_block,n_res_channel=n_res_channel,level="top")


        self.quantizer_t = VectorQuantizer(num_embeddings=self.num_embeddings, embedding_dim=latent_dim,name="top_quantizer")
        self.quantizer_b = VectorQuantizer(num_embeddings=self.num_embeddings, embedding_dim=latent_dim,name="bottom_quantizer")


        self.conditional_bottom=self.build_conditional_bottom()


        ### DECODER PART


        self.decoder=self.build_decoder()

        ## vq vae

        self.vqvae=self.get_vqvae2()

        ## metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")


    def call(self,x):
        """Follows algorithm 1 definition"""

        e_top,e_bottom=self.encode(x)

        recon_img=self.decode([e_top,e_bottom])



        return recon_img




    def build_encoder(self,input_dim,channels,n_res_block,n_res_channel,level="bottom"):

        """

        Rationale:

            The bottom encoder reduce the dimensions until a certain point (for example downsample for a factor 4
            and the top encoder downsample further the image, so we can have two codebook rapresentations


        :param in_channel: int, number of input channel
        :param channel: int, number of channel for the Conv2D layers, half before relu activation and half after
        :param n_res_block: int, number of residual blocks
        :param n_res_channel: int, number of channels in the residual blocks
        :param stride: int 2 or 4 for bottom and top encoder models
        :return:
        """
        if level=="bottom":

            input=Input(shape=input_dim)
            x=Conv2D(channels,kernel_size=4,strides=2,padding="same")(input)
            x=Activation("relu")(x)
            x=Conv2D(n_res_channel, kernel_size=4, strides=2, padding="same")(x)

            for _ in range(n_res_block):
                x=ResidualBlock(n_res_channel)(x)

            model=Model(input,x,name="bottom_encoder")
            return model

        if level=="top":
            "divide the input dim by four and substitute the number of channels because of bottom learner"
            input_shape=(input_dim[0]//4,input_dim[1]//4,n_res_channel)
            input=Input(shape=input_shape)

            x=Conv2D(channels,kernel_size=4,strides=2,padding="same")(input)
            x=Activation("relu")(x)
            x = Conv2D(n_res_channel, kernel_size=4, padding="same")(x)

            for _ in range(n_res_block):
                x=ResidualBlock(n_res_channel)(x)

            model=Model(input,x,name="top_encoder")
            return model




    def build_conditional_bottom(self):
        """
        This layer produce a conditioned output of encoded bottom
        :return Residual block that is able to conditionate with the priors

        """
        "divide the input dim by four and substitute the number of channels because of bottom learner"


        input= Input(shape=(self.input_dim[0] // 4, self.input_dim[1] // 4, self.latent_dim))
        cond_input  = Input(shape=(self.input_dim[0] // 8, self.input_dim[1] // 8, self.latent_dim))

        x=Conv2DTranspose(self.channels,strides=(2,2),kernel_size=3,padding="same")(cond_input)
        merge = Concatenate()([input, x])
        x=Conv2DTranspose(self.latent_dim,kernel_size=3,padding="same")(merge)

        model=Model([input,cond_input],x,name="conditional_bottom")

        return model


    def build_decoder(self):


        e_top_input=Input(shape=(self.input_dim[0]//8,self.input_dim[1]//8,self.latent_dim))                        #smaller codebook
        e_bottom_input=Input(shape=(self.input_dim[0]//4,self.input_dim[1]//4,self.latent_dim))       #bottom codebook

        ## enlarge top inputs

        x_top=Conv2DTranspose(self.channels, 3, strides=2, padding="same")(e_top_input)

        #merge

        merge=Concatenate()([x_top,e_bottom_input])


        x = Conv2DTranspose(self.channels, 3, strides=2, padding="same")(merge)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(self.channels//2, 3, strides=2, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        decoder_outputs = Conv2DTranspose(self.outchannels, 3, activation="sigmoid", padding="same")(x)
        decoder = Model([e_top_input,e_bottom_input], decoder_outputs, name="decoder")

        return decoder


    def decode(self,e):
        """

        :param e: list of form [e_top,e_bottom] where e are embeddings vectors
        :return: reconstructed image
        """
        return self.decoder(e)

    def encode(self,x):

        if isinstance(x,tuple):
            x,_=x

        h_top = self.encoder_t(self.encoder_b(x))  # data flows until the top
        e_top = self.quantizer_t(h_top)

        h_bottom_conditioned = self.conditional_bottom([self.encoder_b(x), e_top])

        e_bottom = self.quantizer_b(h_bottom_conditioned)

        return e_top,e_bottom



    def get_vqvae2(self):

        inputs=self.encoder_b.inputs

        quantized_latents = self.encode(inputs)
        reconstructions = self.decode(quantized_latents)

        model=keras.Model(inputs, reconstructions, name="vq_vae2")
        return model


    def train_step(self, x):
        """

        :param x: images
        :return: metrics
        """
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.



            reconstructions = self.vqvae(x)          #call the vq vae, which records also the commitment and the codebook losses

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


    def save_dict(self,path):
        dictionary_data={"input_dim":self.input_dim, "latent_dim":self.latent_dim, "num_embeddings":self.num_embeddings,"train_variance":self.train_variance,"channels":self.channels,"n_res_block":self.n_res_block,"n_res_channel":self.n_res_channel, "outchannels":self.outchannels}
        a_file = open(path, "w")
        json.dump(dictionary_data, a_file)
        a_file.close()


    def load_weights(self,weights):
        self.encoder_t.load_weights(weights["encoder_t"])
        self.encoder_b.load_weights(weights["encoder_b"])
        self.decoder.load_weights(weights["decoder"])
        self.conditional_bottom.load_weights(weights["conditional_bottom"])

        self.quantizer_t.set_weights((np.load(weights["embeddings_top"])))
        self.quantizer_b.set_weights((np.load(weights["embeddings_bottom"])))
