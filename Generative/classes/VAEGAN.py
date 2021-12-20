import tensorflow as tf
import tensorflow.keras as keras

from .Architectures import ConditionalDiscriminator, Discriminator
from .ConditionalVAE import CVAE
from .VAE import VAE
from tensorflow.keras.layers import *

class VAEGAN(keras.Model):
    """A class to train a VAE-GAN model

    This implementation follows the basic idea of a GAN, so we will have

    VAE:
    Encoder
    Decoder

    GAN
    Decoder/Generator
    Discriminator

    the models share the Decoder weights and are trained in a loop:

    Discriminator trained on real/sampled images with right labels

    VAE trained to reconstruct images

    Sampled images passed to discriminator as real to train the generator to foolish it


    """
    def __init__(self,input_dim,latent_dim,output_channels=1,encoder_architecture=[(0,128),[(0,256)]], decoder_architecture=[(0,128),[(0,256)]],discriminator_architecture=[(0,128),[(0,256)]],discriminator_dense=None):
        """

        :param input_dim: tuple, input dimension (for example (28,28,1)
        :param latent_dim: int, dimension of the latent space
        :param output_channels: number of output channels. Usally should be the same of input_dim[-1]

        """


        super().__init__()

        self.input_dim=input_dim
        self.latent_dim=latent_dim


        self.output_channels=input_dim[-1]


        self.encoder_architecture=encoder_architecture
        self.decoder_architecture=decoder_architecture
        self.discriminator_architecture=discriminator_architecture

        self.discrminator_dense = discriminator_dense
        self.vae = VAE(input_dim, latent_dim,encoder_architecture=encoder_architecture, decoder_architecture=decoder_architecture,)
        self.discriminator = Discriminator(input_shape=input_dim,n_classes=1,
                                                      conv_layer_list=discriminator_architecture,
                                                      dense=discriminator_dense)

#        self.vae = VAE(input_dim, latent_dim)
 #       self.discriminator=self.build_discriminator()
        self.vae.build(input_shape=(None,*input_dim))

    def build_discriminator(self):
        """
        discriminator model

        :return: discriminator model
        """

        # Create the discriminator
        discriminator = keras.Sequential(
            [
                keras.Input(shape=self.input_dim),
                Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                LeakyReLU(alpha=0.2),
                GlobalMaxPooling2D(),
                Dense(1),
            ],
            name="discriminator",
        )
        return discriminator

    def compile(self,vae_opt=keras.optimizers.Adam(),discriminator_optimizer=keras.optimizers.Adam()):
        """
        This method compile both the vae and the discriminator, including the definition
        of the BinaryCrossentropy loss function for the discriminator and the accuracy.

        :param vae_opt: optimizer for variational autoencoder, default Adam
        :param discriminator_optimizer: optimizer for discriminator, default Adam
        """
        super(VAEGAN, self).compile(run_eagerly=True)
        self.vae.compile(optimizer=vae_opt)
        self.discriminator.compile(optimizer=discriminator_optimizer,loss="binary_crossentropy",metrics="accuracy")

        self.vae_optimizer=vae_opt
        self.d_optimizer=discriminator_optimizer
        self.loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
        self.d_accuracy=keras.metrics.Accuracy()

    def train_step(self, data):
        """
        Method used to override the behaviour of .fit method

        This training step involve three different steps:

        step 1: Discriminator training

        Real images and fake ones (generated by noise from the vae's decoder) are passed to the discriminator
        with 0 as labels for real ones and 1 for the fake ones.
        The parameters of the discriminator are updated from the binarycrossentropy computations

        Step 2: VAE training

        Real images x are reconstructed by the VAE as x'=D(E(x)) where E is the encoder network and D the decoder
        Using the KL loss and the reconstruction loss (MSE) weights of the E and D are updated

        Step 3: VAE-GAN training

        Sampled images are passed to the discriminator labelled as real ones with misleading labels
        The Decoder weights are updated using the BinaryCrossentropy gradients.
        This will incourage the decoder to learn how to foolish the discriminator.




        :param data: images passed
        :return: dict with metrics
        """

        ## three step

        batch_size = tf.shape(data)[0]

        # fake 1, real 0

        ## step 1 train the discriminator with real and fake imgs

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.vae.decode(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, data], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(labels.shape)

        # train the discriminator

        with tf.GradientTape() as tape:

            y_pred=self.discriminator(combined_images)

            d_loss=self.loss_fn(labels,y_pred)
            d_acc=self.d_accuracy(tf.round(labels),tf.round(y_pred))

        grads=tape.gradient(d_loss,self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads,self.discriminator.trainable_weights))


        # 2 train the vae

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.vae.encoder(data)
            reconstruction = self.vae.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_vae_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_vae_loss, self.vae.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads, self.trainable_weights))


        #3. sample image and train the decoder in adversial way

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            y_pred = self.discriminator(self.vae.decoder(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, y_pred)

        grads = tape.gradient(g_loss, self.vae.decoder.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads, self.vae.decoder.trainable_weights))

        return {"d_acc":d_acc,"d_loss":d_loss,"vae_loss":total_vae_loss,"kl_loss":kl_loss}


    def __call__(self, x):
        return self.discriminator(self.vae(x))



class CVAEGAN(keras.Model):
    """A class to train a Conditional VAE-GAN model

    This implementation follows the basic idea of a GAN, so we will have

    VAE:
    Encoder
    Decoder

    GAN
    Decoder/Generator
    Discriminator

    the models share the Decoder weights and are trained in a loop:

    Discriminator trained on real/sampled images with right labels

    VAE trained to reconstruct images

    Sampled images passed to discriminator as real to train the generator to foolish it


    """
    def __init__(self,input_dim,latent_dim,output_channels=1,n_classes=10,conditional_shape=(1,),n_emb=50,encoder_architecture=[(0,128),[(0,256)]], decoder_architecture=[(0,128),[(0,256)]],discriminator_architecture=[(0,128),[(0,256)]],discriminator_dense=None):
        """

        :param input_dim: tuple, input dimension (for example (28,28,1)
        :param latent_dim: int, dimension of the latent space
        :param output_channels: number of output channels. Usally should be the same of input_dim[-1]
        :param n_classes: int number of possible classes
        :param conditional_shape: tuple dimension of the conditional input
        :param encoder_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..]
        :param decoder_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..]

        """


        super().__init__()

        self.input_dim=input_dim
        self.latent_dim=latent_dim
        self.output_channels=output_channels
        self.encoder_architecture=encoder_architecture
        self.decoder_architecture=decoder_architecture
        self.discriminator_architecture=discriminator_architecture
        self.conditional_shape=conditional_shape
        self.n_emb=n_emb
        self.n_classes=n_classes
        self.discrminator_dense=discriminator_dense
        self.vae = CVAE(input_dim, latent_dim,n_classes=n_classes,emb_dim=n_emb,encoder_architecture=encoder_architecture,decoder_architecture=decoder_architecture,conditional_shape=conditional_shape)
        self.discriminator=ConditionalDiscriminator(input_shape=input_dim,conditional_shape=conditional_shape,embedding_dim=n_emb,n_classes=1,conv_layer_list=discriminator_architecture,dense=discriminator_dense)
        #self.discriminator=self.build_discriminator()
        #self.vae.build(input_shape=[(None,*input_dim),(None,*conditional_shape)])

    def build_discriminator(self):
        """
        discriminator model

        :return: discriminator model
        """

        # Create the discriminator
        discriminator = keras.Sequential(
            [
                keras.Input(shape=self.input_dim),
                Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                LeakyReLU(alpha=0.2),
                GlobalMaxPooling2D(),
                Dense(1),
            ],
            name="discriminator",
        )
        return discriminator

    def compile(self,vae_opt=keras.optimizers.Adam(),discriminator_optimizer=keras.optimizers.Adam()):
        """
        This method compile both the vae and the discriminator, including the definition
        of the BinaryCrossentropy loss function for the discriminator and the accuracy.

        :param vae_opt: optimizer for variational autoencoder, default Adam
        :param discriminator_optimizer: optimizer for discriminator, default Adam
        """
        super(CVAEGAN, self).compile(run_eagerly=True)
        self.vae.compile(optimizer=vae_opt)
        self.discriminator.compile(optimizer=discriminator_optimizer,loss="binary_crossentropy",metrics="accuracy")

        self.vae_optimizer=vae_opt
        self.d_optimizer=discriminator_optimizer
        self.loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
        self.d_accuracy=keras.metrics.Accuracy()

    def train_step(self, data):
        """
        Method used to override the behaviour of .fit method

        This training step involve three different steps:

        step 1: Discriminator training

        Real images and fake ones (generated by noise from the vae's decoder) are passed to the discriminator
        with 0 as labels for real ones and 1 for the fake ones.
        The parameters of the discriminator are updated from the binarycrossentropy computations

        Step 2: VAE training

        Real images x are reconstructed by the VAE as x'=D(E(x)) where E is the encoder network and D the decoder
        Using the KL loss and the reconstruction loss (MSE) weights of the E and D are updated

        Step 3: VAE-GAN training

        Sampled images are passed to the discriminator labelled as real ones with misleading labels
        The Decoder weights are updated using the BinaryCrossentropy gradients.
        This will incourage the decoder to learn how to foolish the discriminator.




        :param data: images passed
        :return: dict with metrics
        """

        img, conditions = data

        ## three step

        batch_size = tf.shape(img)[0]

        # fake 1, real 0

        ## step 1 train the discriminator with real and fake imgs

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_conditions = tf.random.uniform(shape=[batch_size,*self.conditional_shape],minval=0,maxval=self.n_classes,dtype=tf.int32)


        # Decode them to fake images
        generated_images = self.vae.decode(random_latent_vectors,random_conditions)

        # Combine them with real images
        combined_images = tf.concat([generated_images, img], axis=0)

        combined_conditions=tf.concat([random_conditions,conditions],axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(labels.shape)

        # train the discriminator

        with tf.GradientTape() as tape:

            y_pred=self.discriminator([combined_images,combined_conditions])

            d_loss=self.loss_fn(labels,y_pred)
            d_acc=self.d_accuracy(tf.round(labels),tf.round(y_pred))

        grads=tape.gradient(d_loss,self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads,self.discriminator.trainable_weights))


        # 2 train the vae

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.vae.encoder([img,conditions])
            reconstruction = self.vae.decoder([z,conditions])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(img, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_vae_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_vae_loss, self.vae.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads, self.trainable_weights))


        #3. sample image and train the decoder in adversial way

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_conditions = tf.random.uniform(shape=[batch_size,*self.conditional_shape],minval=0,maxval=self.n_classes,dtype=tf.int32)

        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            y_pred = self.discriminator([self.vae.decoder([random_latent_vectors,random_conditions]),random_conditions])
            g_loss = self.loss_fn(misleading_labels, y_pred)

        grads = tape.gradient(g_loss, self.vae.decoder.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads, self.vae.decoder.trainable_weights))

        return {"d_acc":d_acc,"d_loss":d_loss,"vae_loss":total_vae_loss,"kl_loss":kl_loss}
