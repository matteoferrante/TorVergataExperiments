import sys

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from .Architectures import Decoder,Discriminator


class GAN(keras.Model):
    """
    base class for adversarial network learning that extends keras.Model

    """
    def __init__(self,target_shape,latent_dim,encoder_architecture=[(0,128),[(0,256)]], decoder_architecture=[(0,128),[(0,256)]]):
        """

        Attributes
        ----------

        :param latent_dim: dimension of the latent space (i.e the number of random numbers required to generate an image)
        :param target_shape: tuple, shape of the image
        :param discriminator: model
        :param generator : model
        :param latent_dim: dimension of the latent space (i.e the number of random numbers required to generate an image)
        :param encoder_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..] for discriminator
        :param decoder_architecture: list of tuple, len of list is the number of blocks, [(n_block_res,n_filters)..] for generator


        Methods
        ---------
        build_discriminator : build a sequential Keras model to discriminate between real and fake images
        build_generator: build a sequential Keras model to generate images from noise though Conv2DTranspose layers.

        """
        super().__init__()

        self.target_shape = target_shape
        self.latent_dim = latent_dim
        self.encoder_architecture=encoder_architecture
        self.decoder_architecture=decoder_architecture



        #self.discriminator = self.build_discriminator()
        #self.generator = self.build_generator(latent_dim)

        self.discriminator=Discriminator(target_shape,1,conv_layer_list=encoder_architecture)
        self.generator=Decoder(target_shape,latent_dim,decoder_architecture)

        self.gan=Sequential([self.generator,self.discriminator])

    def compile(self, d_optimizer=tf.keras.optimizers.Adam(3e-4), g_optimizer=tf.keras.optimizers.Adam(3e-4), loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),d_accuracy=tf.keras.metrics.Accuracy()):
        """
        method to compile all modules of this model

        :param d_optimizer: optimizer for discriminator
        :param g_optimizer: optimizer for generator
        :param loss_fn:  loss function to use for discriminator and gan
        :param d_accuracy: metric to measure discriminator performances
        """

        super(GAN, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_accuracy=d_accuracy

    def call(self, inputs, training=None, mask=None):

        batch_size=len(inputs)

        print(F"[DEBUG] {batch_size} {inputs}")
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, inputs], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(labels.shape)

        y_pred = self.discriminator(combined_images)

        d_loss = self.loss_fn(labels, y_pred)
        d_acc = self.d_accuracy(tf.round(labels), tf.round(y_pred))

        return {"d_loss":d_loss,"d_acc": d_acc,"gen_imgs":generated_images}




    def train_step(self, data):
        """
        Training step of .fit method.
        It trains the discriminator using a concatenation of real and fake images labelled as 0 and 1 respectively.
        In a second step it train the gan (generator->discriminator) with frozen discriminator weights and images
        generated from noise from generator, using misleading labels (images are fake but labels are 0, to let the
        generator learn how to foolish the discriminator

        :param data: data used for training (i.e the real images)
        :return: a dict of metrics including g_loss, d_loss and d_accuracy
        """

        batch_size = tf.shape(data)[0]

        #fake 1, real 0


        ## step 1 train the discriminator with real and fake imgs

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))


        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, data], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(labels.shape)




        with tf.GradientTape() as tape:

            y_pred=self.discriminator(combined_images)

            d_loss=self.loss_fn(labels,y_pred)
            d_acc=self.d_accuracy(tf.round(labels),tf.round(y_pred))

        grads=tape.gradient(d_loss,self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads,self.discriminator.trainable_weights))


        ## step 2. try to foolish the discriminator with fake images and backpropagate to generator

        random_latent_vectors=tf.random.normal(shape=(batch_size,self.latent_dim))
        misleading_labels=tf.zeros((batch_size,1))

        with tf.GradientTape() as tape:

            y_pred=self.discriminator(self.generator(random_latent_vectors))
            g_loss=self.loss_fn(misleading_labels,y_pred)

        grads=tape.gradient(g_loss,self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads,self.generator.trainable_weights))


        return {"d_loss":d_loss,"g_loss":g_loss, "d_acc":d_acc}



    @staticmethod
    def build_discriminator():
        """
        discriminator model

        :return: discriminator model
        """

        # Create the discriminator
        discriminator = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )
        return discriminator

    @staticmethod
    def build_generator(latent_dim=128):
        """

        :param latent_dim: dimension of the latent space
        :return: generator model
        """
        generator = keras.Sequential(
            [
                keras.Input(shape=(latent_dim,)),
                # We want to generate 128 coefficients to reshape into a 7x7x128 map
                layers.Dense(7 * 7 * 128),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7, 128)),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        return generator


    def get_dict(self):
        dictionary_data = {"target_shape": self.target_shape, "latent_dim": self.latent_dim,
                           "encoder_architecture": self.encoder_architecture,
                           "decoder_architecture": self.decoder_architecture}
        return dictionary_data