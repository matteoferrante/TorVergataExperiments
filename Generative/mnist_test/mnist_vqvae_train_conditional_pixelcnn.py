import itertools
import os
import sys

from classes.VQVAE import VQVAE
from classes.PixelCNN import PixelCNN,TfDistPixelCNN,ConditionalPixelCNN
from utils.callbacks import WandbImagesVQVAE, Save_VQVAE_Weights, Save_PixelCNN_Weights
import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback
import argparse
from os.path import join as opj


ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",default="models/vq_vae_mnist.h5",help="path of trained vq_vae model in phase")

args=ap.parse_args()

wandb.login()


config={"dataset":"mnist", "type":"C_VQ-VAE","phase":"Conditional PixelCNN"}

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


model_path=args.model
print(f"[INFO] Training PixelCNN Autoregressive model with embeddings from {args.model}")

pixel_BS=256



## GENERATE CODEBOOK

model=VQVAE((28,28,1),latent_dim=16,num_embeddings=128,train_variance=4)

weights=map_models_weights(args.model)

print(weights)
model.load_weights(weights["encoder"],weights["embeddings"],weights["decoder"])




# Generate the codebook indices.
train_dataset_pixel = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset_pixel = train_dataset_pixel.batch(pixel_BS).shuffle(1024)


## predict on batch
encoded_outputs=[]
for batch in train_dataset_pixel:
    x,y=batch

    encoded_batch = model.encoder.predict(x)
    encoded_outputs.append((encoded_batch,y))


#flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])

codebook_indices=[]

for batch in encoded_outputs:
    enc_out,condition=batch
    flat_enc_outputs=enc_out.reshape(-1,enc_out.shape[-1])
    code_index = model.vq_layer.get_code_indices(flat_enc_outputs).numpy()
    codebook_indices.append((list(code_index),condition))

## LAVORARE QUA
conditioned_codebook=[]
codebooks=[]
conditions=[]

input_shape=(7,7)

for batch in codebook_indices:
    codebooks+=batch[0]
    conditions+=list(batch[1].numpy())

codebooks=np.array(codebooks)
codebooks=np.reshape(codebooks,(len(conditions),input_shape[0],input_shape[1]))



print(f"[CHECK] {len(codebooks)} {len(conditions)}")



pixel_cnn = ConditionalPixelCNN(input_dim=input_shape, n_embeddings=128, n_residual=3, n_convlayer=2,n_classes=10,cond_emb=50)
opt = keras.optimizers.Adam(3e-3)
pixel_cnn.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt,metrics="accuracy")

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


#dataset generation

train_ds = tf.data.Dataset.from_tensor_slices(((codebooks,conditions),codebooks))
train_ds = train_ds.shuffle(buffer_size=1024).batch(BS).repeat()

steps=len(codebooks)//BS


pixel_cnn.save_dict("models/vq_vae_conditional_pixelcnn/dict.json")

pixel_cnn.model.fit(train_ds, batch_size=pixel_BS, epochs=30,
              callbacks=callbacks,steps_per_epoch=steps)
pixel_cnn.model.save_weights("models/vq_vae_conditional_pixelcnn/final_conditional_pixelcnn_mnist_vqvae.h5")









