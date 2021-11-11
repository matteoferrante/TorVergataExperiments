import argparse
import json

from imutils import build_montages
from keras import Input

from classes.PixelCNN import PixelCNN
from classes.VQVAE import VQVAE
from utils.functions import  map_vqvae_weights,pixelcnn_sample_vqvae
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

config={"dataset":"mnist", "type":"VQ-VAE","phase":"sample"}

wandb.init(project="TorVergataExperiment-Generative",config=config)


ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",help="path of vq_vae weights")
ap.add_argument("-p","--pixel",help="path of pixel_cnn weights")

args=ap.parse_args()



print(f"[INFO] Loading VQ_VAE Weights")

vqvae = VQVAE((28, 28, 1), latent_dim=16, num_embeddings=128, train_variance=4)

weights = map_vqvae_weights(args.model)

print(weights)
vqvae.load_weights(weights["encoder"], weights["embeddings"], weights["decoder"])


print(f"[INFO] Loading PixelCNN Model")


config=glob(opj(args.pixel,"*.json"))[0]
weights=glob(opj(args.pixel,"*.h5"))[0]

print(f"[CONFIG] PixelCNN {config}")
f=open(config,"r")

config=json.load(f)

input_dim=config["input_dim"]
n_emb=config["num_embeddings"]
n_res=config["n_residual"]
n_conv=config["n_convlayer"]
ksize=config["ksize"]
p=PixelCNN(input_dim=input_dim,n_embeddings=n_emb,n_residual=n_res,n_convlayer=n_conv,ksize=ksize)

p.model.load_weights(weights)

#p.load_model(r"models\vq_vae_pixelcnn\final_pixelcnn_mnist_vqvae.h5")

print(f"[SUMMARY] {p.model.summary()}")

#p.model.set_weights(weights)

#ps=PixelCNNSampler(p,vqvae=model)



print(f"[INFO] Sampling")

generated_samples=pixelcnn_sample_vqvae(100,p.model,vqvae)

images = generated_samples * 255.
images = np.repeat(images, 3, axis=-1)
vis = build_montages(images, (28, 28), (10, 10))[0]

log = {f"image_sampled": wandb.Image(vis)}
wandb.log(log)

