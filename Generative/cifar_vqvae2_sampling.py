import argparse
import json

from imutils import build_montages
from keras import Input

from classes.PixelCNN2 import ConditionalPixelCNN2
from classes.VQVAE import VQVAE2
from utils.functions import map_vqvae_weights, pixelcnn_sample_vqvae, map_vqvae2_weights, pixelcnn_sample_vqvae2
from tensorflow.keras.models import model_from_json
from glob import glob
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow import keras
import tensorflow as tf
import wandb

wandb.login()

config={"dataset":"cifar", "type":"VQ-VAE-2","phase":"sample"}

wandb.init(project="TorVergataExperiment-Generative",config=config)

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",help="path of vq_vae weights")
ap.add_argument("-p","--pixel",help="path of pixel_cnn weights")

args=ap.parse_args()

print(f"[INFO] Retrieve VQ VAE")

model_path=args.model

pixel_BS=256

jdict_path=glob(opj(model_path,"*.json"))[0]

with open(jdict_path) as f:
    jdict=json.load(f)

model=VQVAE2(**jdict)

weights=map_vqvae2_weights(model_path)
model.load_weights(weights)

print(f"[INFO] Retrieve PixelCNN2")


config=glob(opj(args.pixel,"*.json"))[0]
p_weights=glob(opj(args.pixel,"*.h5"))[0]

print(f"[CONFIG] PixelCNN {config}")
f=open(config,"r")

config=json.load(f)

p=ConditionalPixelCNN2(**config)
p.model.load_weights(p_weights)




print(f"[INFO] Sampling")

conditions=np.repeat(np.arange(0,10),10)


##TODO lavorare qua su shape output
generated_samples=pixelcnn_sample_vqvae2(100,input_dim=(4,4),bottom_dim=(8,8),conditions=conditions,pixel_cnn=p.model,vqvae=model)

images = generated_samples * 255.
#images = np.repeat(images, 3, axis=-1)
vis = build_montages(images, (28, 28), (10, 10))[0]

log = {f"image_sampled": wandb.Image(vis)}
wandb.log(log)

