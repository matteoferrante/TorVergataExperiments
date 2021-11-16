import glob
import itertools
import json
import os
import sys

import tqdm
from tensorflow.keras.models import load_model

from classes.VQVAE import VQVAE2
from classes.PixelCNN2 import ConditionalPixelCNN2
from utils.callbacks import WandbImagesVQVAE, Save_VQVAE_Weights, Save_PixelCNN_Weights, Save_VQVAE2_Weights, \
    WandbImagesVQVAE2
from utils.functions import map_vqvae2_weights

import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback
import argparse
from os.path import join as opj


ap=argparse.ArgumentParser()
ap.add_argument("-p","--phase",default=0,help="phase of training. Phase 0 is VQ_VAE Traininig, Phase 1 is PixelCNN training",type=int)
ap.add_argument("-m","--model",default="models/vq_vae_mnist.h5",help="path of trained vq_vae model in phase 0, useful for phase 1")

args=ap.parse_args()

wandb.login()

if args.phase==0:
    phase="VQ_VAE2_Training"

elif args.phase==1:
    phase="PixelCNN2_Training"

config={"dataset":"cifar", "type":"VQ-VAE_2","phase":phase}

wandb.init(project="TorVergataExperiment-Generative",config=config)




def map_models_weights(model_dir):
    files=os.listdir(model_dir)
    d={}
    for f in files:
        if "encoder" in f:
            d["encoder"]=opj(model_dir,f)
        elif ("generator" in f) or ("decoder" in f):
            d["decoder"]=opj(model_dir,f)
        elif "embeddings" in f:
            d["embeddings"]=opj(model_dir,f)
    return d





## DATA
BS = 256

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

ts = len(x_train) // BS
vs = len(x_test) // BS

x_train=x_train.astype("float32")/255.
x_test=x_test.astype("float32")/255.

#x_train = np.expand_dims(x_train.astype("float32") / 255., -1)
#x_test = np.expand_dims(x_test.astype("float32") / 255., -1)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BS).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.shuffle(1024).batch(BS).repeat()

##CHECKPOINT




if args.phase==0:
    print(f"[INFO] Training VQ_VAE Model")


    g=VQVAE2((32,32,3),latent_dim=16,num_embeddings=128,train_variance=4,n_res_channel=16)






    model_check= Save_VQVAE2_Weights(output_dir="models",outname="vq_vae2",endname="cifar10")



    es=tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=3,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )


    callbacks=[
        WandbImagesVQVAE2(test_dataset,sample=False),
        WandbCallback(),
        model_check,
        es,
    ]

    ### TRAINING

    g.compile(keras.optimizers.Adam())
    g.fit(train_dataset,validation_data=test_dataset,steps_per_epoch=ts,validation_steps=vs,epochs=20,callbacks=callbacks)

    g.save_dict("models/vq_vae2/dict.json")
    g.vqvae.save_weights("models/vq_vae2/model_vqvae2_weights_cifar10.h5")
    g.vqvae.save("models/vq_vae2/model_vqvae2_model_cifar10.h5")

elif args.phase==1:
    model_path=args.model
    print(f"[INFO] Training PixelCNN Autoregressive model with embeddings from {args.model}")

    pixel_BS=256

    jdict_path=glob.glob(opj(model_path,"*.json"))[0]

    with open(jdict_path) as f:
        jdict=json.load(f)
    print(f"[PARAMETERS] {jdict_path}")

    ## GENERATE CODEBOOK

    model=VQVAE2(**jdict)

    weights=map_vqvae2_weights(model_path)


    model.load_weights(weights)

    ## LAVORARE QUA

    # Generate the codebook indices.

    train_dataset_pixel = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset_pixel = train_dataset_pixel.batch(pixel_BS).shuffle(1024, seed=42)

    e_top_list = []
    e_bottom_list = []
    y_list = []

    start = True
    for batch in tqdm.tqdm(train_dataset_pixel):
        x, y = batch

        e_top, e_bottom = model.encode(x)
        if start:
            top_shape = (e_top.shape[1], e_top.shape[2])

            bottom_shape = (e_bottom.shape[1], e_bottom.shape[2])
            start = False

        e_top = e_top.numpy().reshape(-1, e_top.shape[-1])
        e_top = model.quantizer_t.get_code_indices(e_top).numpy()

        e_bottom = e_bottom.numpy().reshape(-1, e_bottom.shape[-1])
        e_bottom = model.quantizer_b.get_code_indices(e_bottom).numpy()

        e_top_list.append(list(e_top))
        e_bottom_list.append(list(e_bottom))

        y_list.append(y)

    codebook_top_indices = list(itertools.chain.from_iterable(e_top_list))

    codebook_bottom_indices = list(itertools.chain.from_iterable(e_bottom_list))
    y_list = list(itertools.chain.from_iterable(y_list))

    codebook_top_indices = np.array(codebook_top_indices)
    codebook_top_indices = np.reshape(codebook_top_indices, (len(x_train), top_shape[0], top_shape[1]))

    codebook_bottom_indices = np.array(codebook_bottom_indices)
    codebook_bottom_indices = np.reshape(codebook_bottom_indices, (len(x_train), bottom_shape[0], bottom_shape[1]))

    print(f"Shape of the training data for PixelCNN: {codebook_top_indices.shape},{codebook_bottom_indices.shape}")



    pixel_cnn = ConditionalPixelCNN2(input_top_dim=top_shape,input_bottom_dim=bottom_shape,n_embeddings=128, n_residual=10, n_convlayer=2)
    opt = keras.optimizers.Adam(3e-3)
    pixel_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt,metrics="accuracy")

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    callbacks = [WandbCallback(), es]


    pixel_cnn.save_dict("models/vq_vae2_conditional_pixelcnn/dict.json")

    pixel_cnn.fit((codebook_top_indices, codebook_bottom_indices, np.array(y_list)),
                  (codebook_top_indices, codebook_bottom_indices), batch_size=pixel_BS, validation_split=0.1, epochs=30,
                  callbacks=callbacks)
    pixel_cnn.model.save_weights("models/vq_vae2_conditional_pixelcnn/final_pixelcnn2_cifar_vqvae2.h5")












