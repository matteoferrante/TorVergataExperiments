"""Same as Progressive Growing GAN but conditional"""


import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import *
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K



class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        """Layer used in fading phase while growing"""
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"alpha": self.alpha})
        return config


class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(input_shape[-1],), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        return inputs + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = super().get_config().copy()
        return config

class MinibatchStdev(Layer):
    """layer that help to contrast model collapse, computing statistics also coming from the minibatch"""
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        combined = tf.concat([inputs, minibatch_stddev], axis=-1)

        return combined

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


class WeightScaling(Layer):
    """Weights initialization"""
    def __init__(self, shape, gain=np.sqrt(2), **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain * tf.math.rsqrt(fan_in)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * self.wscale

    def compute_output_shape(self, input_shape):
        return input_shape


def WeightScalingDense(x, filters, gain, use_pixelnorm=False, activate=None):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = K.int_shape(x)[-1]
    x = Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate == 'LeakyReLU':
        x = LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = Activation('tanh')(x)

    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x


def WeightScalingConv(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1, 1)):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = K.int_shape(x)[-1]
    x = Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init,
                      dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate == 'LeakyReLU':
        x = LeakyReLU(0.2)(x)
    elif activate == 'tanh':
        x = Activation('tanh')(x)
    elif activate == 'sigmoid':
        x = Activation('sigmoid')(x)

    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x


class PixelNormalization(Layer):
    """This normalize each convolutional layer of the generator"""

    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        normalized = inputs * l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape



class CPGGAN(keras.Model):

    """Conditional Progressive growing GANs based on https://github.com/fabulousjeong/pggan-tensorflow/blob/main/pgan.py"""


    def __init__(self, start_dim=4,latent_dim=100,n_block=4,outchannels=3 ,d_steps=1,gp_weight=10.0,drift_weight=0.001,num_classes=10,emb_dim=50):
        super().__init__()

        self.start_dim=(start_dim,start_dim,outchannels)
        self.latent_dim=latent_dim
        self.n_block=n_block
        self.outchannels = outchannels
        self.d_steps=d_steps
        self.gp_weight=gp_weight
        self.drift_weight=drift_weight
        self.num_classes=num_classes
        self.emb_dim=emb_dim

        self.filters= [512, 512, 512, 512, 256, 128, 64]

        self.discriminator=self.init_discriminator()
        self.discriminator_wt_fade=None
        self.generator=self.init_generator()
        self.n_depth=0


    def init_generator(self):
        noise = Input(shape=(self.latent_dim,))

        in_label=Input(shape=(1,))                                  #input for condition the class
        li=Embedding(self.n_classes,self.emb_dim)(in_label)              #li stands for label input
        li=Dense(self.latent_dim)(li)
        #li = Reshape(input_shape)(li)                              #produce image compatible shapes



        merge=Concatenate()[noise,li]

        x = PixelNormalization()(merge)
        # Actual size(After doing reshape) is just FILTERS[0], so divide gain by 4
        x = WeightScalingDense(x, filters=self.start_dim[0]*self.start_dim[1]*self.filters[0], gain=np.sqrt(2)/4, activate='LeakyReLU', use_pixelnorm=True)
        x = Reshape((self.start_dim[0], self.start_dim[0], self.filters[0]))(x)

        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)

        # Add "toRGB", the original paper uses linear as actiavation.
        # Gain should be 1, cos it's a last layer
        x = WeightScalingConv(x, filters=3, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False)

        g_model = Model([noise,in_label], x, name='generator')
        g_model.summary()
        return g_model

    def init_discriminator(self):

        ## label input

        in_label=Input(shape=(1,))                  #input for condition the class
        li=Embedding(self.n_classes,self.emb_dim)(in_label)              #li stands for label input
        li=Dense(np.prod(self.start_dim))(li)
        li = Reshape(self.start_dim)(li)        #produce image compatible shapes


        img_input = Input(shape=self.start_dim,dtype=tf.float32)

        merge=Concatenate()[img_input,li]

        #img_input = tf.cast(img_input, tf.float32)
        # fromRGB
        x = WeightScalingConv(merge, filters=self.filters[0], kernel_size=(1, 1), gain=np.sqrt(2), activate='LeakyReLU')
        # Add Minibatch end of discriminator
        x = MinibatchStdev()(x)

        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(3, 3), gain=np.sqrt(2), activate='LeakyReLU')
        x = WeightScalingConv(x, filters=self.filters[0], kernel_size=(4, 4), gain=np.sqrt(2), activate='LeakyReLU',
                              strides=(4, 4))

        x = Flatten()(x)
        # Gain should be 1, cos it's a last layer
        x = WeightScalingDense(x, filters=1, gain=1.)

        d_model = Model([img_input,in_label], x, name='discriminator')

        return d_model

    # Fade in upper resolution block
    def fade_in_generator(self):

        """Fade in generator block to gradually weight the output of the new layer"""

        # for layer in self.generator.layers:
        #    layer.trainable = False
        # 1. Get the node above the “toRGB” block
        block_end = self.generator.layers[-5].output
        # 2. Double block_end
        block_end = UpSampling2D((2, 2))(block_end)

        # 3. Reuse the existing “toRGB” block defined as“x1”.
        x1 = self.generator.layers[-4](block_end)  # Conv2d
        x1 = self.generator.layers[-3](x1)  # WeightScalingLayer
        x1 = self.generator.layers[-2](x1)  # Bias
        x1 = self.generator.layers[-1](x1)  # tanh

        # 4. Define a "fade in" block (x2) with two 3x3 convolutions and a new "toRGB".
        x2 = WeightScalingConv(block_end, filters=self.filters[self.n_depth], kernel_size=(3, 3), gain=np.sqrt(2),
                               activate='LeakyReLU', use_pixelnorm=True)
        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth], kernel_size=(3, 3), gain=np.sqrt(2),
                               activate='LeakyReLU', use_pixelnorm=True)

        x2 = WeightScalingConv(x2, filters=3, kernel_size=(1, 1), gain=1., activate='sigmoid', use_pixelnorm=False)

        # Define stabilized(c. state) generator
        self.generator_stabilize = Model(self.generator.input, x2, name='generator')

        # 5.Then "WeightedSum" x1 and x2 to smoothly put the "fade in" block.
        x = WeightedSum()([x1, x2])
        self.generator = Model(self.generator.input, x, name='generator')

        self.generator.summary()



    # Fade in upper resolution block
    def fade_in_discriminator(self):
        #for layer in self.discriminator.layers:
        #    layer.trainable = False
        input_shape = list(self.discriminator.input.shape)
        # 1. Double the input resolution.
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3])
        img_input = Input(shape = input_shape,dtype=tf.float32)

        in_label=Input(shape=(1,))                  #input for condition the class
        li=Embedding(self.n_classes,self.emb_dim)(in_label)              #li stands for label input
        li=Dense(np.prod(input_shape))(li)
        li = Reshape(input_shape)(li)        #produce image compatible shapes



        merge=Concatenate()[img_input,li]


        #img_input = tf.cast(img_input, tf.float32)

        # 2. Add pooling layer
        #    Reuse the existing “formRGB” block defined as “x1".
        x1 = AveragePooling2D()(merge)
        x1 = self.discriminator.layers[1](x1) # Conv2D FromRGB
        x1 = self.discriminator.layers[2](x1) # WeightScalingLayer
        x1 = self.discriminator.layers[3](x1) # Bias
        x1 = self.discriminator.layers[4](x1) # LeakyReLU

        # 3.  Define a "fade in" block (x2) with a new "fromRGB" and two 3x3 convolutions.
        #     Add an AveragePooling2D layer
        x2 = WeightScalingConv(img_input, filters=self.filters[self.n_depth], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU')

        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = WeightScalingConv(x2, filters=self.filters[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')

        x2 = AveragePooling2D()(x2)

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block.
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator
        for i in range(5, len(self.discriminator.layers)):
            x2 = self.discriminator.layers[i](x2)
        self.discriminator_stabilize = Model([img_input,in_label], x2, name='discriminator')

        # 5. Add existing discriminator layers.
        for i in range(5, len(self.discriminator.layers)):
            x = self.discriminator.layers[i](x)
        self.discriminator = Model([img_input,in_label], x, name='discriminator')

        self.discriminator.summary()


    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize
        self.discriminator.summary()

    # Change to stabilized(c. state) generator
    def stabilize_generator(self):
        self.generator = self.generator_stabilize

        self.generator.summary()



    def compile(self, d_optimizer, g_optimizer):
        super(CPGGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)

                # Calculate the drift for regularization
                drift = tf.reduce_mean(tf.square(real_logits))

                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + self.gp_weight * gp + self.drift_weight * drift

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = -tf.reduce_mean(gen_img_logits)
        # Get the gradients w.r.t the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))
        return {'d_loss': d_loss, 'g_loss': g_loss}