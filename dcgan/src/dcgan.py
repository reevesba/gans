import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam

class DCGAN():
    def __init__(self):
        # input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        # input image dimensions
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # size of noise vector, used as input to generator
        self.z_dim = 100

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

        # build the generator
        self.generator = self.build_generator()

        # keep descriminator's parameters constant for generator training
        self.discriminator.trainable = False

        self.gan = self.build_gan()
        self.gan.compile(loss="binary_crossentropy", optimizer=Adam())

        # variables for plotting
        self.losses = []
        self.accuracies = []
        self.sample_intervals = []

    def build_generator(self):
        model = Sequential()

        # reshape into 7x7x256 tensor via fully connected layer
        model.add(Dense(256*7*7, input_dim=self.z_dim))
        model.add(Reshape((7, 7, 256)))

        # transposed conv layer, from 7x7x256 to 14x14x128 tensor
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # transposed conv layer, from 14x14x28 to 14x14x64 tensor
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # transposed conv layer, from 14x14x64 to 28x28x1 tensor
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        return model

    def build_discriminator(self):
        model = Sequential()

        # conv layer, from 28x28x1 to 14x14x32 tensor
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.01))

        # conv layer, from 14x14x32 to 7x7x64 tensor
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # conv layer, from 7x7x64 to 3x3x128 tensor
        model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # output layer with sigmoid activation
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        return model

    def build_gan(self):
        model = Sequential()

        # combine generator and discriminator
        model.add(self.generator)
        model.add(self.discriminator)

        model.summary()
        return model

    def train(self, epochs, batch_size=128, sample_interval=50):
        # load MNIST dataset
        (X_train, _), (_, _) = mnist.load_data()

        # rescale [0, 255] grayscale pixel values to [-1,  1]
        X_train = X_train/127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)

        # real image labels: all ones
        real = np.ones((batch_size, 1))

        # fake image labels: all zeros
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # get random batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # generate batch of fake images
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(z)

            # train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5*np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # generate new batch of fake images
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(z)

            # train the generator (wants discriminator to mistake images as real)
            g_loss = self.gan.train_on_batch(z, real)

            # if at save interval => save generated image samples
            if (epoch + 1) % sample_interval == 0:
                # save losses and accuracies for plotting
                self.losses.append((d_loss, g_loss))
                self.accuracies.append(100.0*accuracy)
                self.sample_intervals.append(epoch + 1)
              
                # output training progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch + 1, d_loss, 100.0*accuracy, g_loss))
                self.sample_images(epoch + 1)

    def sample_images(self, epoch, grid_rows=4, grid_cols=4):
        # generate fake images from random noise
        z = np.random.normal(0, 1, (grid_rows*grid_cols, self.z_dim))
        gen_imgs = self.generator.predict(z)

        # rescale image pixel values to [0, 1]
        gen_imgs = 0.5*gen_imgs + 0.5

        # set image grid
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(4, 4), sharey=True, sharex=True)

        cnt = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("out/generated_digits/mnist_%d.png" % epoch)
        plt.close()

    def plot_losses(self):
        losses = np.array(self.losses)

        # plot training losses for discriminator and generator
        plt.figure(figsize=(15, 5))
        plt.plot(self.save_intervals, losses.T[0], label="Discriminator loss")
        plt.plot(self.save_intervals, losses.T[1], label="Generator loss")

        plt.xticks(self.save_intervals, rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("out/losses_fig.png")

    def plot_accuracies(self):
        accuracies = np.array(self.accuracies)

        # plot discriminator accuracy
        plt.figure(figsize=(15, 5))
        plt.plot(self.save_intervals, accuracies, label="Discriminator accuracy")

        plt.xticks(self.save_intervals, rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig("out/accuracies_fig.png")

def main():
    # set hyperparameters
    epochs = 1000
    batch_size = 128
    sample_interval = 100

    # execute dcgan model
    dcgan = DCGAN()
    dcgan.train(epochs, batch_size, sample_interval)
    dcgan.plot_losses()
    dcgan.plot_accuracies()

if __name__ == '__main__':
    main()

