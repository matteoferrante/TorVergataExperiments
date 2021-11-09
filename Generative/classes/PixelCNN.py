import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow import keras

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

class PixelCNN:


    """This is a class for autoregressive model like pixelCNN
    PixelCNN is an autoregressive model that paramtetrize the predicted value of a pixel i with the previous value 0,1,i-1
    This implementation is based on https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PixelCNN

    As alternative is possible to follow: https://keras.io/examples/generative/pixelcnn/ for an implementation with keras
    """

    def __init__(self,input_dim=(28,28,1),num_resnet=1,num_hierarchies=2,num_filters=32,num_logistic_mix=5,dropout_p=.3):
        self.input_dim=input_dim
        self.num_resnet=num_resnet
        self.num_hierarchies=num_hierarchies
        self.num_filters=num_filters
        self.num_logistic_mix=num_logistic_mix
        self.dropout_p=dropout_p


    def build_model(self):
        self.dist=tfp.distributions.PixelCNN(image_shape=self.input_dim,num_resnet=self.num_resnet,num_hierarchies=self.num_hierarchies,num_filters=self.num_filters,num_logistic_mix=self.num_logistic_mix,dropout_p=self.dropout_p)

        self.input=tfkl.Input(self.input_dim)
        self.log_prob=self.dist.log_prob(self.input)


        self.model=Model(inputs=self.input,outputs=self.log_prob)
        self.model.add_loss(-tf.reduce_mean(self.log_prob))

    def compile(self,optimizer=keras.optimizers.Adam(),metrics=[]):
        self.model.compile(optimizer=optimizer,metrics=metrics)

    def fit(self,data,epochs=10,batch_size=128,validation_split=0.1):
        self.model.fit(data,data,batch_size=batch_size,epochs=epochs,validation_split=validation_split)

    def sample(self,n_images):
        return self.dist.sample(n_images)