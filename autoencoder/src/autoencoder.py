from __future__ import print_function
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

batch_size = 100
original_dim = 28*28
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

def sampling(args):
    # sample from latent space
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var/2)*epsilon

# write the encoder

# input to our encoder
x = Input(shape=(original_dim,), name="input")
# intermediate layer
h = Dense(intermediate_dim, activation="relu", name="encoding")(x)
# mean of latent space
z_mean = Dense(latent_dim, name="mean")(h)
# log variance of latent space
z_log_var = Dense(latent_dim, name="log-variance")(h)
# note that output_shape isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# define encoder as a keras model
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")

# write the decoder

# input to the decoder
input_decoder = Input(shape=(latent_dim,), name="decoder_input")
# takes latent space to intermediate dimension
decoder_h = Dense(intermediate_dim, activation="relu", name="decoder_h")(input_decoder)
# get mean from original dimension
x_decoded = Dense(original_dim, activation="sigmoid", name="flat_decoded")(decoder_h)
# define decoder as a keras model
decoder = Model(input_decoder, x_decoded, name="decoder")

# combine encoder and decoder into VAE model

# grab the output 'sampling z'
output_combined = decoder(encoder(x)[2])
# link input and overall output
vae = Model(x, output_combined)

def vae_loss(x=tf.Tensor, x_decoded_mean=tf.Tensor, z_log_var=z_log_var, z_mean=z_mean, original_dim=original_dim):
    # compare each grayscale pixel x to the value in x_decoded_mean
    xent_loss = original_dim*objectives.binary_crossentropy(x, x_decoded_mean)
    # KL divergence: measures the difference between two distributions
    kl_loss = - 0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # lump loss measurements together
    return xent_loss + kl_loss

# compile VAE model
vae.compile(optimizer="rmsprop", loss=vae_loss)
vae.summary()

# create standard train/test split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# finally, train the model
vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test), verbose=1)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test, cmap="viridis")
plt.colorbar()
plt.savefig("out/latent_space_classes")

# display a 2D manifold of the digits
n = 15	# figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size*n, digit_size*n))

# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i*digit_size:(i + 1)*digit_size, j*digit_size:(j + 1)*digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="Greys_r")
plt.savefig("out/generate_digits.png")
