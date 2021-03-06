import os
from os.path import join as opj
import numpy as np
import tensorflow_probability as tfp
import tqdm
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf


def map_vqvae_weights(model_dir):
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

def map_vqvae2_weights(model_dir):
    files=os.listdir(model_dir)
    d={}
    for f in files:
        if "encoder_b" in f:
            d["encoder_b"]=opj(model_dir,f)
        elif "encoder_t" in f:
            d["encoder_t"] = opj(model_dir, f)
        elif "conditional_bottom" in f:
            d["conditional_bottom"]=opj(model_dir,f)
        elif ("generator" in f) or ("decoder" in f):
            d["decoder"]=opj(model_dir,f)
        elif "embeddings_bottom" in f:
            d["embeddings_bottom"]=opj(model_dir,f)
        elif "embeddings_top" in f:
            d["embeddings_top"]=opj(model_dir,f)
    return d




def pixelcnn_sample_vqvae(n,pixel_cnn,vqvae):
    # Create a mini sampler model.
    inputs = Input(shape=pixel_cnn.input_shape[1:])
    x = pixel_cnn(inputs, training=False)
    dist = tfp.distributions.Categorical(logits=x)
    sampled = dist.sample()
    sampler = keras.Model(inputs, sampled)

    # Create an empty array of priors.
    batch = n
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    pretrained_embeddings = vqvae.vq_layer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(vqvae.encoder.output.shape[1:])))

    # Generate novel images.

    generated_samples = vqvae.decoder.predict(quantized)
    return generated_samples



def pixelcnn_sample_c_vqvae(n,conditions,input_dim,pixel_cnn,vqvae):

    """samples from conditional pixelcnn to vqvae"""

    # Create a mini sampler model.
    inputs = Input(shape=input_dim)
    conditions_input=Input(shape=(1,))
    x = pixel_cnn([inputs,conditions_input], training=False)
    dist = tfp.distributions.Categorical(logits=x)
    sampled = dist.sample()
    sampler = keras.Model([inputs,conditions_input], sampled)

    # Create an empty array of priors.
    batch = n
    priors = np.zeros(shape=(batch,input_dim[0],input_dim[1]))
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict([priors,conditions])
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    pretrained_embeddings = vqvae.vq_layer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(vqvae.encoder.output.shape[1:])))

    # Generate novel images.

    generated_samples = vqvae.decoder.predict(quantized)
    return generated_samples



def pixelcnn_sample_vqvae2(n,conditions,input_dim,bottom_dim,pixel_cnn,vqvae):

    """samples from conditional pixelcnn to vqvae2"""

    # Create a mini sampler model.
    top_inputs = Input(shape=input_dim)
    bottom_inputs=Input(shape=bottom_dim)
    conditions_input=Input(shape=(1,))
    x,y = pixel_cnn([top_inputs,bottom_inputs,conditions_input], training=False)
    dist_x = tfp.distributions.Categorical(logits=x)
    dist_y = tfp.distributions.Categorical(logits=y)

    sampled_x = dist_x.sample()
    sampled_y = dist_y.sample()

    sampler = keras.Model([top_inputs,bottom_inputs,conditions_input], [sampled_x,sampled_y])

    # Create an empty array of priors.
    batch = n
    top_priors = np.zeros(shape=(batch,input_dim[0],input_dim[1]),dtype=int)
    bottom_priors=np.zeros(shape=(batch,bottom_dim[0],bottom_dim[1]),dtype=int)
    batch, b_rows, b_cols = bottom_priors.shape
    batch, t_rows, t_cols = top_priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in tqdm.tqdm(range(b_rows)):
        for col in range(b_cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            for t_row in range(t_rows):
                for t_col in range(t_cols):

                   # print(f"top_row: {t_row} top_col: {t_col} bottom_row: {row} bottom_col {col}")
                    #print("BEFORE",top_priors.shape,bottom_priors.shape,conditions.shape)
                    top_probs,bottom_probs=sampler.predict([top_priors,bottom_priors,conditions])

                    #print("OUTPUT", top_probs.shape, bottom_probs.shape)

                    # Use the probabilities to pick pixel values and append the values to the priors.
                    top_priors[:,t_row,t_col]=top_probs[:,t_row,t_col]
                    bottom_priors[:,row,col]=bottom_probs[:,row,col]



    pretrained_top_embeddings = vqvae.quantizer_t.embeddings
    pretrained_bottom_embeddings = vqvae.quantizer_b.embeddings

    priors_top_ohe = tf.one_hot(top_priors.astype("int32"), vqvae.num_embeddings).numpy()
    priors_bottom_ohe = tf.one_hot(bottom_priors.astype("int32"), vqvae.num_embeddings).numpy()

    quantized_top = tf.matmul(
        priors_top_ohe.astype("float32"), pretrained_top_embeddings, transpose_b=True
    )
    quantized_top = tf.reshape(quantized_top, (-1, *(vqvae.encoder_t.output.shape[1:])))

    quantized_bottom= tf.matmul(
        priors_bottom_ohe.astype("float32"), pretrained_bottom_embeddings, transpose_b=True
    )
    quantized_bottom = tf.reshape(quantized_bottom, (-1, *(vqvae.encoder_b.output.shape[1:])))

    # Generate novel images.

    generated_samples = vqvae.decoder.predict([quantized_top,quantized_bottom])
    return generated_samples