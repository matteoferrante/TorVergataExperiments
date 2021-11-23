import itertools
import os
import sys

from classes.VQVAE import VQVAE
from classes.PixelCNN import PixelCNN,TfDistPixelCNN
from utils.callbacks import WandbImagesVQVAE, Save_VQVAE_Weights, Save_PixelCNN_Weights
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
    phase="VQ_VAE_Training"

elif args.phase==1:
    phase="PixelCNN_Training"

config={"dataset":"mnist", "type":"VQ-VAE","phase":phase}

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

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

ts = len(x_train) // BS
vs = len(x_test) // BS

x_train = np.expand_dims(x_train.astype("float32") / 255., -1)
x_test = np.expand_dims(x_test.astype("float32") / 255., -1)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BS).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.shuffle(1024).batch(BS).repeat()

##CHECKPOINT




if args.phase==0:
    print(f"[INFO] Training VQ_VAE Model")


    g=VQVAE((28,28,1),latent_dim=16,num_embeddings=128,train_variance=4)

    print(g.encoder.summary())

    print(g.decoder.summary())





    model_check= Save_VQVAE_Weights(output_dir="../models", outname="vq_vae", endname="mnist")



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
        WandbImagesVQVAE(test_dataset,sample=False),
        WandbCallback(),
        model_check,
        es,
    ]

    ### TRAINING

    g.compile(keras.optimizers.Adam())
    g.fit(train_dataset,validation_data=test_dataset,steps_per_epoch=ts,validation_steps=vs,epochs=5,callbacks=callbacks)


elif args.phase==1:
    model_path=args.model
    print(f"[INFO] Training PixelCNN Autoregressive model with embeddings from {args.model}")

    pixel_BS=256



    ## GENERATE CODEBOOK

    model=VQVAE((28,28,1),latent_dim=16,num_embeddings=128,train_variance=4)

    weights=map_models_weights(args.model)

    print(weights)
    model.load_weights(weights["encoder"],weights["embeddings"],weights["decoder"])




    # Generate the codebook indices.
    train_dataset_pixel = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset_pixel = train_dataset_pixel.batch(pixel_BS).shuffle(1024)

    encoded_outputs = model.encoder.predict(train_dataset_pixel)
    #flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

    codebook_indices=[]
    for b in range(len(train_dataset_pixel)):

        #flat the outputs
        flat_enc_outputs=encoded_outputs[b*pixel_BS:(b+1)*pixel_BS].reshape(-1,encoded_outputs.shape[-1])
        code_index=model.vq_layer.get_code_indices(flat_enc_outputs).numpy()
        codebook_indices.append(list(code_index))

    codebook_indices = list(itertools.chain.from_iterable(codebook_indices))
    codebook_indices = np.array(codebook_indices)
    print(codebook_indices.shape)
    codebook_indices = codebook_indices.reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")


    input_shape=(codebook_indices.shape[1],codebook_indices.shape[2])

    pixel_cnn = PixelCNN(input_dim=input_shape, n_embeddings=128, n_residual=3, n_convlayer=2)
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


    pixel_cnn.save_dict("models/vq_vae_pixelcnn/dict")

    pixel_cnn.fit(codebook_indices, codebook_indices, batch_size=pixel_BS, validation_split=0.1, epochs=30,
                  callbacks=callbacks)
    pixel_cnn.model.save_weights("models/vq_vae_pixelcnn/final_pixelcnn_mnist_vqvae.h5")











    ## RIFACCIO CON TFDISTPIXELCNN

    ## PROVO CN TFPIXELCNN

    # tfpix = TfDistPixelCNN(num_hierarchies=1, num_filters=64)
    # tfpix.model.summary()
    # tfpix.compile()
    #
    # pixel_BS = 1
    #
    # codebook_ds = tf.data.Dataset.from_tensor_slices((codebook_indices, codebook_indices))
    # codebook_ds = codebook_ds.repeat().shuffle(1024).batch(pixel_BS)
    #
    # train_step_pixel = len(codebook_indices) // pixel_BS
    #
    # tfpix.model.fit(codebook_ds, steps_per_epoch=train_step_pixel)
    #
    # tfpix.model.save("models/vq_vae_pixelcnn/tfdistpixel_cnn_mnist.h5")



