import os
import sys

from classes.VAE import VAE
from utils.callbacks import WandbImagesVAE, SaveGeneratorWeights, SaveVAEWeights
import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback


wandb.login()



config={"dataset":"mnist", "type":"VAE"}

wandb.init(project="TorVergataExperiment-Generative",config=config)

## DATA


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()



###


encoder_architecture=[(1,64),(1,128),(1,256),(1,512)]
decoder_architecture=[(1,64),(1,128),(1,256),(1,512)]

BS=256
g=VAE((28,28,1),latent_dim=100)

print(g.encoder.summary())

print(g.decoder.summary())


ts=len(x_train)//BS
vs=len(x_test)//BS

x_train=np.expand_dims(x_train.astype("float32")/255.,-1)
x_test=np.expand_dims(x_test.astype("float32")/255.,-1)



train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset=  train_dataset.shuffle(buffer_size=1024).batch(BS).repeat()


test_dataset=tf.data.Dataset.from_tensor_slices(x_test)
test_dataset=test_dataset.shuffle(1024).batch(BS).repeat()

##CHECKPOINT

#model_check=SaveGeneratorWeights(filepath="models/generator_vae_mnist.h5")

os.makedirs("models/vae",exist_ok=True)
model_check=SaveVAEWeights(filepath="models/vae")


callbacks=[
    WandbImagesVAE(test_dataset),
    WandbCallback(),
    model_check,
]

### TRAINING

g.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
g.fit(train_dataset,validation_data=test_dataset,steps_per_epoch=ts,validation_steps=vs,epochs=40,callbacks=callbacks)

