from classes.GAN import GAN
from utils.callbacks import WandbImagesGAN, SaveGeneratorWeights
import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback


wandb.login()

config={"dataset":"mnist","type":"GAN"}

wandb.init(project="TorVergataExperiment-Generative",config=config)

## DATA


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()



###
BS=256
g=GAN(BS)


ts=len(x_train)//BS

x_train=np.expand_dims(x_train.astype("float32")/255.,-1)
x_test=np.expand_dims(x_test.astype("float32")/255.,-1)



train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset=  train_dataset.shuffle(buffer_size=1024).batch(BS).repeat()


test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataset=test_dataset.shuffle(1024).batch(BS)

##CHECKPOINT

model_check=SaveGeneratorWeights(filepath="models/generator_gan_mnist.h5")



callbacks=[
    WandbImagesGAN(),
    WandbCallback(),
    model_check
]

### TRAINING

g.compile()
g.fit(train_dataset,steps_per_epoch=ts,epochs=40,callbacks=callbacks)

