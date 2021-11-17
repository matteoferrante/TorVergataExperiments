import os

import tensorflow as tf
import numpy as np
import glob
import argparse
from classes.PGGAN import PGGAN
from utils.callbacks import WandbImagesPGGAN
import wandb
import tensorflow.keras as keras
from os.path import join as opj
from wandb.keras import WandbCallback


wandb.login()

checkpoint_path="models/PGGAN"
config={"dataset":"cifar", "type":"PG-GAN"}

wandb.init(project="TorVergataExperiment-Generative",config=config)


BS_list = [256,128,64,32]

BS=BS_list[0]
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

ts = len(x_train) // BS
vs = len(x_test) // BS

x_train=x_train.astype("float32")/255.
x_test=x_test.astype("float32")/255.


NOISE_DIM = 128
# Set the number of batches, epochs and steps for trainining.
# Look 800k images(16x50x1000) per each lavel
EPOCHS_PER_RES = 1


## INIT

def resize(img,target_size=(4,4)):
    return tf.image.resize(img,target_size)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BS).repeat().map(resize)

# Instantiate the optimizer for both networks
# learning_rate will be equalized per each layers by the WeightScaling scheme
generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

pgan = PGGAN(
    latent_dim = NOISE_DIM,
    d_steps = 1,
)

callbacks=[WandbImagesPGGAN(),WandbCallback()]

pgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

os.makedirs(checkpoint_path,exist_ok=True)
# Start training the initial generator and discriminator
pgan.fit(train_dataset, steps_per_epoch = ts, epochs = EPOCHS_PER_RES, callbacks=callbacks)
pgan.save_weights(opj(checkpoint_path, f"checkpoint_path_ndepth_{pgan.n_depth}_weights_cifar.h5"))

tf.keras.utils.plot_model(pgan.generator, to_file=opj(checkpoint_path,f'generator_{pgan.n_depth}.png'), show_shapes=True)
tf.keras.utils.plot_model(pgan.discriminator, to_file=opj(checkpoint_path,f'discriminator_{pgan.n_depth}.png'), show_shapes=True)


# Train faded-in / stabilized generators and discriminators
for n_depth in range(1, 4):



  print(f"[INFO] Fading phase for {n_depth}")
  # Set current level(depth)
  pgan.n_depth = n_depth

  new_dim=(pgan.n_depth+1)*4
  new_dim=(new_dim,new_dim)

  ##dataset redefinition
  BS=BS_list[n_depth]
  ts = len(x_train) // BS
  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)

  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BS).repeat().map(lambda  x: resize(x,new_dim))

  #enlarge network

  pgan.fade_in_generator()
  pgan.fade_in_discriminator()

  # Draw fade in generator and discriminator
  tf.keras.utils.plot_model(pgan.generator, to_file=opj(checkpoint_path,f'generator_{pgan.n_depth}.png'), show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=opj(checkpoint_path,f'discriminator_{pgan.n_depth}.png'), show_shapes=True)

  pgan.compile(
      d_optimizer=discriminator_optimizer,
      g_optimizer=generator_optimizer,
  )
  # Train fade in generator and discriminator
  pgan.fit(train_dataset, steps_per_epoch=ts, epochs=EPOCHS_PER_RES, callbacks=callbacks)
  pgan.save_weights(opj(checkpoint_path, f"checkpoint_path_ndepth_{n_depth}_weights_cifar.h5"))


  print(f"[INFO] Stabilizing phase for {n_depth}")
  pgan.stabilize_generator()
  pgan.stabilize_discriminator()

  # Draw fade in generator and discriminator
  tf.keras.utils.plot_model(pgan.generator, to_file=opj(checkpoint_path,f'generator_{pgan.n_depth}_stabilized.png'), show_shapes=True)
  tf.keras.utils.plot_model(pgan.discriminator, to_file=opj(checkpoint_path,f'discriminator_{pgan.n_depth}_stabilized.png'), show_shapes=True)

  pgan.compile(d_optimizer=discriminator_optimizer,g_optimizer=generator_optimizer,)
  # Train stabilized generator and discriminator
  pgan.fit(train_dataset, steps_per_epoch = ts, epochs = EPOCHS_PER_RES, callbacks=callbacks)
  pgan.save_weights(opj(checkpoint_path, f"checkpoint_path_ndepth_{pgan.n_depth}_stabilized_weights_cifar.h5"))
