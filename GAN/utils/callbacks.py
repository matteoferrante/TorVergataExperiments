import tensorflow as tf
import tensorflow.keras as  keras
import wandb
import numpy as np
from imutils import build_montages

class WandbImages(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb
    """

    def on_epoch_end(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(256, self.model.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)

        images = ((generated_images * 127.5) + 127.5)
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (16, 16))[0]

        log={f"image":wandb.Image(vis)}
        wandb.log(log)



class WandbImagesConditional(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb
    """

    def on_epoch_end(self, epoch, logs=None):

        n_classes=self.model.n_classes
        conditions=np.repeat(np.arange(0,n_classes,1).tolist(),10)
        random_latent_vectors = tf.random.normal(shape=(n_classes*10, self.model.latent_dim))
        generated_images = self.model.generator([random_latent_vectors,conditions])

        images = generated_images *255.
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (n_classes, 10))[0]

        log={f"image":wandb.Image(vis)}
        wandb.log(log)


class SaveGeneratorWeights(keras.callbacks.Callback):

    def __init__(self, filepath):
        super().__init__()
        self.filepath=filepath

    """A custom callback to save generator weights"""


    def on_epoch_end(self, epoch,logs=None):
        self.model.generator.save_weights(self.filepath)