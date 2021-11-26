import sys


sys.path.append(".")

import json
import tensorflow as tf
import tensorflow.keras as  keras
import wandb
import numpy as np
from imutils import build_montages
import os
from os.path import join as opj


class WandbImagesGAN(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb
    """

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape=target_shape

    def on_epoch_end(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(100, self.model.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)

        images = generated_images.numpy() * 255
        if self.target_shape[-1]==1:
            images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (self.target_shape[0], self.target_shape[1]), (10, 10))[0]

        log={f"image":wandb.Image(vis)}
        wandb.log(log)



class WandbImagesConditionalGAN(keras.callbacks.Callback):
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



class WandbImagesPGGAN(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb
    """

    def on_epoch_end(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(100, self.model.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        images = generated_images.numpy() * 255.


        #images = np.repeat(images, 3, axis=-1)

        act_dim=4*(self.model.n_depth+1)

        vis = build_montages(images, (act_dim,act_dim), (10, 10))[0]

        log={f"image_sampled{self.model.n_depth}":wandb.Image(vis)}
        wandb.log(log)

class WandbImagesCPGGAN(keras.callbacks.Callback):

    #TODO fix random conditions with fixed conditions!

    """
    A custom Callback to produce a grid of images in wandb
    """

    def on_epoch_end(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(100, self.model.latent_dim))
        #conditions = np.repeat(np.arange(0, self.model.num_classes, 1).tolist(), 10)

        random_conditions = tf.random.uniform(shape=[100, ], minval=0, maxval=self.model.num_classes, dtype=tf.int32)

        generated_images = self.model.generator([random_latent_vectors,random_conditions])
        images = generated_images.numpy() * 255.


        #images = np.repeat(images, 3, axis=-1)

        act_dim=4*(self.model.n_depth+1)

        vis = build_montages(images, (act_dim,act_dim), (10, 10))[0]

        log={f"image_sampled{self.model.n_depth}":wandb.Image(vis)}
        wandb.log(log)




class WandbImagesVAE(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb for VAE
    """

    def __init__(self, validation_data,target_shape):

        """Workaround to keep validation data!"""

        super().__init__()
        self.validation_data = validation_data
        self.target_shape=target_shape

    def on_epoch_end(self, epoch, logs=None):


        if self.validation_data:
            x_recon=self.model(next(iter(self.validation_data)))

            x_recon=x_recon[:100] ## use more than 100 in bS
            images = x_recon.numpy() * 255.
            if self.target_shape[-1]==1:

                images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (self.target_shape[0], self.target_shape[1]), (10, 10))[0]

            log={f"image":wandb.Image(vis)}
            wandb.log(log)
        else:
            print(f"No validation data {self.validation_data}")

        ## just sampling

        z=np.random.randn(100,self.model.latent_dim)
        x_sampled=self.model.decode(z)

        images = x_sampled.numpy() *255.

        if self.target_shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (self.target_shape[0], self.target_shape[1]), (10, 10))[0]

        log = {f"image_sampled": wandb.Image(vis)}
        wandb.log(log)


class WandbImagesCVAE(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb for VAE
    """

    def __init__(self, validation_data):

        """Workaround to keep validation data!"""

        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):


        if self.validation_data:
            x_recon=self.model(next(iter(self.validation_data)))

            x_recon=x_recon[:100] ## use more than 100 in bS
            images = x_recon.numpy() * 255.
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (10, 10))[0]

            log={f"image":wandb.Image(vis)}
            wandb.log(log)
        else:
            print(f"No validation data {self.validation_data}")

        ## just sampling

        z=np.random.randn(100,self.model.latent_dim)
        n_classes = self.model.n_classes
        conditions = np.repeat(np.arange(0, n_classes, 1).tolist(), 10)
        x_sampled=self.model.decode(z,conditions)

        images = x_sampled.numpy() *255.
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (10, 10))[0]

        log = {f"image_sampled": wandb.Image(vis)}
        wandb.log(log)

class WandbImagesVAEGAN(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb for VAE
    """

    def __init__(self, validation_data):

        """Workaround to keep validation data!"""

        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):


        if self.validation_data:
            x_recon=self.model.vae(next(iter(self.validation_data)))

            x_recon=x_recon[:100] ## use more than 100 in bS
            images = x_recon.numpy() * 255.
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (10, 10))[0]

            log={f"image":wandb.Image(vis)}
            wandb.log(log)
        else:
            print(f"No validation data {self.validation_data}")

        ## just sampling

        z=np.random.randn(100,self.model.latent_dim)
        x_sampled=self.model.vae.decode(z)

        images = x_sampled.numpy() *255.
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (10, 10))[0]

        log = {f"image_sampled": wandb.Image(vis)}
        wandb.log(log)


class SaveGeneratorWeights(keras.callbacks.Callback):

    def __init__(self, filepath):
        super().__init__()
        self.filepath=filepath

    """A custom callback to save generator weights"""


    def on_epoch_end(self, epoch,logs=None):

        try:
            self.model.generator.save_weights(self.filepath)
        except:
            self.model.encoder.save_weights(self.filepath)


class SaveVAEWeights(keras.callbacks.Callback):

    def __init__(self,filepath):
        self.filepath=filepath
        super().__init__()

    def on_train_begin(self, logs=None):
        """
        Create the directory and save some info about the architecture

        """
        os.makedirs(self.filepath,exist_ok=True)
        tf.keras.utils.plot_model(
            self.model,
            to_file=opj(self.filepath,"vae_model.png"),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )


    def on_epoch_end(self, epoch,logs=None):

        self.model.encoder.save_weights(opj(self.filepath,"encoder_weights.h5"))
        self.model.decoder.save_weights(opj(self.filepath,"decoder_weights.h5"))


class SaveGANWeights(keras.callbacks.Callback):

    def __init__(self,filepath):
        self.filepath=filepath
        super().__init__()

    def on_train_begin(self, logs=None):
        """
        Create the directory and save some info about the architecture

        """
        os.makedirs(self.filepath,exist_ok=True)
        tf.keras.utils.plot_model(
            self.model,
            to_file=opj(self.filepath,"vae_model.png"),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )


    def on_epoch_end(self, epoch,logs=None):

        self.model.generator.save_weights(opj(self.filepath,"decoder_weights.h5"))
        self.model.discriminator.save_weights(opj(self.filepath,"discriminator_weights.h5"))


class SaveVAEGANWeights(keras.callbacks.Callback):

    def __init__(self, filepath):
        super().__init__()
        self.filepath=filepath

    """A custom callback to save VAEGAN weights"""


    def on_epoch_end(self, epoch,logs=None):

        self.model.vae.decoder.save_weights(opj(self.filepath,"decoder_weights.h5"))
        self.model.vae.encoder.save_weights(opj(self.filepath,"encoder_weights.h5"))
        self.model.discriminator.save_weights(opj(self.filepath,"discriminator_weights.h5"))


class Save_VQVAE_Weights(keras.callbacks.Callback):


    def __init__(self,filepath):
        self.filepath=filepath
        super().__init__()

    def on_train_begin(self, logs=None):
        """
        Create the directory and save some info about the architecture

        """
        os.makedirs(self.filepath,exist_ok=True)
        tf.keras.utils.plot_model(
            self.model,
            to_file=opj(self.filepath,"vqvae_model.png"),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

    def on_epoch_end(self, epoch,logs=None):

        self.model.encoder.save_weights(opj(self.filepath,f"vq_vae_encoder.h5"))

        self.model.decoder.save_weights(opj(self.filepath,f"vq_vae_decoder.h5"))


    def on_train_end(self, logs=None):
        emb=self.model.vq_layer.get_weights()
        np.save(opj(self.filepath,f"vq_vae_embeddings.npy"),emb)




class Save_PixelCNN_Weights(keras.callbacks.Callback):

    def __init__(self, output_dir,outname,endname="mnist"):
        """

        :param output_dir: name of the output directory
        :param outname: name of subdirectory to create used to save encoder, emebeddings and decoder
        :param endname: usually the name of the dataset, used as end part of the saved name.
        (for example end name: mnist -> files are saved as encoder_mnist.h5, embeddings_mnist.h5 and decoder_mnist.h5)
        """
        super().__init__()
        self.outdir=opj(output_dir,outname)
        self.endname=endname
        os.makedirs(self.outdir,exist_ok=True)


    def on_epoch_end(self, epoch,logs=None):

        self.model.save_weights(opj(self.outdir,f"pixel_cnn_{self.endname}.h5"))



    def on_train_begin(self, save_also_config=False,logs=None):
        if save_also_config:
            config=self.model.get_config()
            a_file = open(opj(self.outdir,f"pixel_cnn_config_{self.endname}.json"), "w")
            json.dump(config, a_file)
            a_file.close()

        json_dict=self.model.to_json()
        with open(opj(self.outdir,f"pixel_cnn_{self.endname}.json"), 'w', encoding='utf-8') as f:
            json.dump(json_dict, f)



class Save_VQVAE2_Weights(keras.callbacks.Callback):
    """Class to save vqvae2 weights"""

    def __init__(self, output_dir,outname,endname="mnist"):
        """

        :param output_dir: name of the output directory
        :param outname: name of subdirectory to create used to save encoder, emebeddings and decoder
        :param endname: usually the name of the dataset, used as end part of the saved name.
        (for example end name: mnist -> files are saved as encoder_mnist.h5, embeddings_mnist.h5 and decoder_mnist.h5)
        """
        super().__init__()
        self.outdir=opj(output_dir,outname)
        self.endname=endname
        os.makedirs(self.outdir,exist_ok=True)


    def on_epoch_end(self, epoch,logs=None):

        self.model.encoder_b.save_weights(opj(self.outdir,f"vq_vae2_encoder_b_{self.endname}.h5"))
        self.model.encoder_t.save_weights(opj(self.outdir, f"vq_vae2_encoder_t_{self.endname}.h5"))

        self.model.conditional_bottom.save_weights(opj(self.outdir, f"vq_vae2_encoder_conditional_bottom_{self.endname}.h5"))

        self.model.decoder.save_weights(opj(self.outdir,f"vq_vae2_decoder_{self.endname}.h5"))


    def on_train_end(self, logs=None):
        emb_b=self.model.quantizer_b.get_weights()

        emb_t=self.model.quantizer_t.get_weights()

        np.save(opj(self.outdir,f"vq_vae_embeddings_bottom_{self.endname}.npy"),emb_b)
        np.save(opj(self.outdir,f"vq_vae_embeddings_top_{self.endname}.npy"),emb_t)







class WandbImagesVQVAE(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb for VAE
    """

    def __init__(self, validation_data,sample=None,pixel_cnn=None):

        """

        :param validation_data: dataset of images, data to reconstruct
        :param sampling:  bool, if true this callback will sample some examples from latent codebook. It requires a pixel_cnn model
        :param pixel_cnn: keras model required to sample the prior. It has to be trained separately
        """
        super().__init__()
        self.validation_data = validation_data
        self.sample=sample
        self.pixel_cnn=pixel_cnn

    def on_epoch_end(self, epoch, logs=None):


        if self.validation_data:
            x_recon=self.model(next(iter(self.validation_data)))

            x_recon=x_recon[:100] ## use more than 100 in bS
            images = x_recon.numpy() * 255.

            if images.shape[-1]==1:
                images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (images.shape[1], images.shape[2]), (10, 10))[0]

            log={f"image":wandb.Image(vis)}
            wandb.log(log)
        else:
            print(f"No validation data {self.validation_data}")

        if self.sample:

            #TODO: use pixelCNN to sample the prior over the codebook
            pass



class WandbImagesVQVAE2(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb for VAE
    """

    def __init__(self, validation_data,sample=None,pixel_cnn=None):

        """

        :param validation_data: dataset of images, data to reconstruct
        """
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):


        if self.validation_data:
            x=next(iter(self.validation_data))
            x_recon=self.model.vqvae(x)

            x_recon=x_recon[:100] ## use more than 100 in bS

            images = x_recon.numpy() * 255.

            #images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (10, 10))[0]

            log={f"image":wandb.Image(vis)}
            wandb.log(log)
        else:
            print(f"No validation data {self.validation_data}")



class WandbVAECallback(keras.callbacks.Callback):
    """
    A custom Callback to produce a grid of images in wandb for VAE
    """


    def on_epoch_end(self, epoch, logs=None):

        log={"loss": self.model.total_loss_tracker.result(),
        "reconstruction_loss": self.model.reconstruction_loss_tracker.result(),
        "kl_loss": self.model.kl_loss_tracker.result()}
        wandb.log(log)




