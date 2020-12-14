import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Dropout, Flatten, Input, Lambda, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

class Dataset:
    def __init__(self, num_labeled):
        # number labeled examples to use for training
        self.num_labeled = num_labeled

        # load the MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        # preprocess the MNIST dataset
        self.x_train = self.preprocess_imgs(self.x_train)
        self.y_train = self.preprocess_labels(self.y_train)
        self.x_test = self.preprocess_imgs(self.x_test)
        self.y_test = self.preprocess_labels(self.y_test)

    def preprocess_imgs(self, x):
        # rescale [0, 255] grayscale pixel values to [-1, 1]
        x = (x.astype(np.float32) - 127.5)/127.5

        # expand image dimensions to width x height x channels
        x = np.expand_dims(x, axis=3)
        return x

    def preprocess_labels(self, y):
        return y.reshape(-1, 1)

    def batch_labeled(self, batch_size):
        # get a random batch of labeled images and their labels
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        # get a random batch of unlabeled images
        idx = np.random.randint(self.num_labeled, self.x_train.shape[0],
                                batch_size)
        imgs = self.x_train[idx]
        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test

class SGAN:

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dim = 100
        self.num_classes = 10
        self.dataset = Dataset(num_labeled=100)

        # core discriminator network:
        # these layers are shared during supervised and unsupervised training
        discriminator_net = self.build_discriminator_net()

        # build & compile the discriminator for supervised training
        self.discriminator_supervised = self.build_discriminator_supervised(discriminator_net)
        self.discriminator_supervised.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

        # build & compile the discriminator for unsupervised training
        self.discriminator_unsupervised = self.build_discriminator_unsupervised(discriminator_net)
        self.discriminator_unsupervised.compile(loss='binary_crossentropy', optimizer=Adam())

        # build the generator
        self.generator = self.build_generator()

        # keep discriminators parameters constant for generator training
        self.discriminator_unsupervised.trainable = False

        # build and compile GAN model with fixed discriminator to train the generator
        # note that we are using the discriminator version with unsupervised output
        self.gan = self.build_gan()
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam())

        # save metrics
        self.supervised_losses = []
        self.iteration_checkpoints = []

        # building fully-supervised classifier for comparison
        self.mnist_classifier = self.build_discriminator_supervised(self.build_discriminator_net())
        self.mnist_classifier.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=Adam())

        # fully supervised metrics
        self.losses = []
        self.accuracies = []

    def build_generator(self):
        model = Sequential()

        # reshape input into 7x7x256 tensor via a fully connected layer
        model.add(Dense(256*7*7, input_dim=self.z_dim))
        model.add(Reshape((7, 7, 256)))

        # transposed convolution layer, from 7x7x256 into 14x14x128 tensor
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # transposed convolution layer, from 14x14x128 to 14x14x64 tensor
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # transposed convolution layer, from 14x14x64 to 28x28x1 tensor
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
        model.add(Activation('tanh'))

        return model

    def build_discriminator_net(self):
        # note: batch normalization doesn't work here in hpc stack
        model = Sequential()

        # convolutional layer, from 28x28x1 into 14x14x32 tensor
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.01))

        # convolutional layer, from 14x14x32 into 7x7x64 tensor
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
        model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # this is where we diverge from ConvNet GAN
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes))

        return model

    def build_discriminator_supervised(self, discriminator_net):
        # predicted probability distribution over real classes
        model = Sequential()

        model.add(discriminator_net)
        model.add(Activation('softmax'))

        return model

    def build_discriminator_unsupervised(self, discriminator_net):
        # 'real-vs-fake' output neuron defined above
        model = Sequential()

        model.add(discriminator_net)

        def predict(x):
            #transform distribution over real classes into binary real-vs-fake probability
            prediction = 1.0 - (1.0/(K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
            return prediction

        model.add(Lambda(predict))

        return model

    def build_gan(self):
        model = Sequential()

        model.add(self.generator)
        model.add(self.discriminator_unsupervised)

        return model

    def train(self, iterations, batch_size, sample_interval):
        # labels for real images: all ones
        real = np.ones((batch_size, 1))

        # labels for fake images: all zeros
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            # -------------------------
            #  Train the Discriminator
            # -------------------------

            # get labeled examples
            imgs, labels = self.dataset.batch_labeled(batch_size)

            # one-hot encode labels
            labels = to_categorical(labels, num_classes=self.num_classes)

            # get unlabeled examples
            imgs_unlabeled = self.dataset.batch_unlabeled(batch_size)

            # generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(z)

            # train on real labeled examples
            d_loss_supervised, accuracy = self.discriminator_supervised.train_on_batch(imgs, labels)

            # train on real unlabeled examples
            d_loss_real = self.discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)

            # train on fake examples
            d_loss_fake = self.discriminator_unsupervised.train_on_batch(gen_imgs, fake)
            d_loss_unsupervised = 0.5*np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train the Generator
            # ---------------------

            # generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(z)

            # train Generator
            g_loss = self.gan.train_on_batch(z, np.ones((batch_size, 1)))

            if (iteration + 1) % sample_interval == 0:
                # save discriminator supervised classification loss to be plotted after training
                self.supervised_losses.append(d_loss_supervised)
                self.iteration_checkpoints.append(iteration + 1)

                # output training progress
                print("%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss unsupervised: %.4f] [G loss: %f]" % (iteration + 1, d_loss_supervised, 100*accuracy, d_loss_unsupervised, g_loss))

    def train_supervised_network(self, batch_size, epochs):
        imgs, labels = self.dataset.training_set()

        # one-hot encode labels
        labels = to_categorical(labels, num_classes=self.num_classes)

        # train the classifier
        training = self.mnist_classifier.fit(x=imgs, y=labels, batch_size=batch_size, epochs=epochs, verbose=1)

        # save metrics
        self.losses = training.history['loss']
        self.accuracies = training.history['acc']

    def plot_losses(self):
        losses = np.array(self.supervised_losses)

        # plot discriminator supervised loss
        plt.figure(figsize=(15, 5))
        plt.plot(self.iteration_checkpoints, losses, label="Discriminator loss")
        plt.xticks(self.iteration_checkpoints, rotation=90)
        plt.title("Discriminator  Supervised Loss")
        plt.xlabel("Iteratiion")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("out/discrminator-loss.png")

    def plot_supervised_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(self.losses), label="Loss")
        plt.title("Classification Loss")
        plt.legend()
        plt.savefig("out/fully-supervised-loss")

    def plot_supervised_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.array(self.accuracies), label="Accuracy")
        plt.title("Classification Accuracy")
        plt.legend()
        plt.savefig("out/fully-supervised-accuracy")

    def train_accuracy(self):
        x, y = self.dataset.training_set()
        y = to_categorical(y, num_classes=self.num_classes)

        # compute classification accuracy on the training set
        _, accuracy = self.discriminator_supervised.evaluate(x, y)
        print("\nTraining Accuracy: %.2f%%" % (100*accuracy))

    def supervised_train_accuracy(self):
        x, y = self.dataset.training_set()
        y = to_categorical(y, num_classes=self.num_classes)

        # compute classification accuracy on the training set
        _, accuracy = self.mnist_classifier.evaluate(x, y)
        print("\nFully Supervised Training Accuracy: %.2f%%" % (100*accuracy))

    def test_accuracy(self):
        x, y = self.dataset.test_set()
        y = to_categorical(y, num_classes=self.num_classes)

        # compute classification accuracy on the test set
        _, accuracy = self.discriminator_supervised.evaluate(x, y)
        print("\nTest Accuracy: %.2f%%" % (100*accuracy))

    def supervised_test_accuracy(self):
        x, y = self.dataset.test_set()
        y = to_categorical(y, num_classes=self.num_classes)

        # compute classification accuracy on the test set
        _, accuracy = self.mnist_classifier.evaluate(x, y)
        print("\nFully Supervised Test Accuracy: %.2f%%" % (100*accuracy))

def main():
    sgan = SGAN()

    # train the SGAN for the specified number of iterations
    sgan.train(iterations=1000, batch_size=32, sample_interval=100)
    sgan.plot_losses()

    # get training and test accuracies
    sgan.train_accuracy()
    sgan.test_accuracy()

    # compare with fully supervised classifier
    sgan.train_supervised_network(batch_size=32, epochs=30)
    sgan.plot_supervised_loss()
    sgan.plot_supervised_accuracy()
    sgan.supervised_train_accuracy()
    sgan.supervised_test_accuracy()

if __name__ == "__main__": 
    main()

