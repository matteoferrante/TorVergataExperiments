import os
import sys

from classes.GAN import GAN
from utils.callbacks import WandbImagesVAE, SaveGeneratorWeights, SaveVAEWeights, WandbVAECallback, WandbImagesGAN, \
    SaveGANWeights
import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback
from imutils import paths
import matplotlib.pyplot as plt


wandb.login()


encoder_architecture=[(1,64),(1,128),(1,256),(1,384),(1,512)]
decoder_architecture=[(1,512),(1,384),(1,256),(1,128),(1,64)]

g=GAN((128,128,3),
      latent_dim=512,
      encoder_architecture=encoder_architecture,
      decoder_architecture=decoder_architecture)


config={"dataset":"celebA", "type":"GAN","encoder_architecture":encoder_architecture,"decoder_architecture":decoder_architecture}
config.update(g.get_dict())


images_dir=r"/home/matteo/NeuroGEN/Dataset/Img/img_align_celeba"

#other important definitions

EPOCHS=50
BS=128
INIT_LR=1e-4

config["epochs"]=EPOCHS
config["BS"]=BS
config["init_lr"]=INIT_LR

wandb.init(project="TorVergataExperiment-Generative",config=config)

def load_images(imagePath):
    # read the image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (128, 128)) / 255.0

    # eventually load other information like attributes here

    # return the image and the extra info

    return image


print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(images_dir))


train_len=int(0.8*len(imagePaths))
val_len=int(0.1*len(imagePaths))
test_len=int(0.1*len(imagePaths))

train_imgs=imagePaths[:train_len]                                #      80% for training
val_imgs=imagePaths[train_len:train_len+val_len]                 #      10% for validation
test_imgs=imagePaths[train_len+val_len:]                         #      10% for testing

print(f"[TRAINING]\t {len(train_imgs)}\n[VALIDATION]\t {len(val_imgs)}\n[TEST]\t\t {len(test_imgs)}")


train_dataset = tf.data.Dataset.from_tensor_slices(train_imgs)
train_dataset = (train_dataset
    .shuffle(1024)
    .map(load_images)
    .cache()
    .repeat()
    .batch(BS)
)

ts=len(train_imgs)//BS

##VALIDATION

val_dataset = tf.data.Dataset.from_tensor_slices(val_imgs)
val_dataset = (val_dataset
    .shuffle(1024)
    .map(load_images)
    .cache()
    .repeat()
    .batch(BS)
)

vs=len(val_imgs)//BS

## TEST

test_dataset = tf.data.Dataset.from_tensor_slices(test_imgs)
test_dataset = (test_dataset
    .shuffle(1024)
    .map(load_images)
    .cache()
    .batch(BS)
)

os.makedirs("models/gan",exist_ok=True)
model_check=SaveGANWeights(filepath="models/gan")

g.compile()


callbacks=[
    WandbImagesGAN(target_shape=(128,128,3)),
    WandbCallback(),
    model_check,
]


g.fit(train_dataset,validation_data=test_dataset,steps_per_epoch=ts,validation_steps=vs,epochs=EPOCHS,callbacks=callbacks)
