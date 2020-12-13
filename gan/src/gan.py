import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1	# because they are grayscale images

# input image dimensions
img_shape = (img_rows, img_cols, channels)

# size of noise vector (generator input)
z_dim = 100

def build_generator(img_shape, z_dim):
    ''' Notes:
        LeakyReLU allows a small positive gradient. Prevents gradients from dying out during training.
        tanh activation scales output images to range [-1, 1], tends to produce crisper images.
    '''
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim))		# fully connected layer
    model.add(LeakyReLU(alpha=0.01))			# Leaky ReLU activation
    model.add(Dense(28*28*1, activation="tanh"))	# output layer w/ tanh activation
    model.add(Reshape(img_shape))			# reshape output to image dimensions
    return model

def build_discriminator(img_shape):
    ''' Notes:
       sigmoid activation ensures output will be in range [0, 1]
    '''
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))		# flatten the input image
    model.add(Dense(128))				# fully connected layer
    model.add(LeakyReLU(alpha=0.01))			# Leaky ReLU activation
    model.add(Dense(1, activation="sigmoid"))		# output layer w/ sigmoid activation
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    # combine generator and discriminator models
    model.add(generator)
    model.add(discriminator)
    return model

# build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

# build the generator
generator = build_generator(img_shape, z_dim)

# keep discriminator's parameters constant for generator training
discriminator.trainable = False

# build and compile GAN w/ fixed discriminator to train generator
gan = build_gan(generator, discriminator)
gan.compile(loss="binary_crossentropy", optimizer=Adam())

def sample_images(generator, iteration, image_grid_rows=4, image_grid_columns=4):
    # sample random noise
    z = np.random.normal(0, 1, (image_grid_rows*image_grid_columns, z_dim))

    # generate images from random noise
    gen_imgs = generator.predict(z)

    # rescale image pixel values to [0, 1]
    gen_imgs = 0.5*gen_imgs + 0.5

    # set image grid
    fix, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
            axs[i, j].axis("off")
            cnt += 1
    plt.savefig("out/generated_digits/generated_digits_" + str(iteration) + ".png")
    plt.close()

# training
losses = []
accuracies = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    # load mnist dataset
    (X_train, _), (_, _) = mnist.load_data()

    # rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train/127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)

    # labels for real images: all ones
    real = np.ones((batch_size, 1))

    # labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        # -----------------------
        # Train the Discriminator
        # -----------------------

        # get a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # train discriminator
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5*np.add(d_loss_real, d_loss_fake)

        # -------------------
        # Train the Generator
        # -------------------

        # generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # train generator
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1)%sample_interval == 0:
            # save losses and accuracies to be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0*accuracy)
            iteration_checkpoints.append(iteration + 1)

            # output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration + 1, d_loss, 100.0*accuracy, g_loss))

            # output a sample of generated images
            sample_images(generator, iteration + 1)

# train the GAN and inspect output

# set hyperparameters
iterations = 20000
batch_size = 128
sample_interval = 1000

# train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)

# plot training losses for discriminator and generator
losses = np.array(losses)

plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

plt.savefig("out/losses.png")

# plot discriminator accuracy
accuracies = np.array(accuracies)

plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.savefig("out/accuracies.png")
