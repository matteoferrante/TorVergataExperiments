import sys

from classes.ConditionalVAE import CVAE
from utils.callbacks import WandbImagesCVAE, SaveGeneratorWeights
import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback


wandb.login()


config={"dataset":"mnist", "type":"C-VAE"}

wandb.init(project="TorVergataExperiment-Generative",config=config)

## DATA


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()



###
BS=256
g=CVAE((28,28,1),100)

print(g.encoder.summary())

print(g.decoder.summary())


ts=len(x_train)//BS
vs=len(x_test)//BS

x_train=np.expand_dims(x_train.astype("float32")/255.,-1)
x_test=np.expand_dims(x_test.astype("float32")/255.,-1)



train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset=  train_dataset.shuffle(buffer_size=1024).batch(BS).repeat()


test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataset=test_dataset.shuffle(1024).batch(BS).repeat()

##CHECKPOINT

model_check=SaveGeneratorWeights(filepath="../models/generator_cvae_mnist.h5")



callbacks=[
    WandbImagesCVAE(test_dataset),
    WandbCallback(),
    model_check,
]

### TRAINING

g.compile()
g.fit(train_dataset,validation_data=test_dataset,steps_per_epoch=ts,validation_steps=vs,epochs=40,callbacks=callbacks)

