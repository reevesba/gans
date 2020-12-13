import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Embedding, Flatten, Input, Multiply, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam

class CGAN:
    
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        # input image dimensions
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # size of the noise vector, used as input to the generator
        self.z_dim = 100

        # number of classes in the dataset
        self.num_classes = 10

        # build and compile the discriminator
        self.discriminator = self.build_cgan_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # build the generator
        self.generator = self.build_cgan_generator()

        # keep discriminator's parameters constant for generator training
        self.discriminator.trainable = False

        # build and compile CGAN model with fixed Discriminator to train the Generator
        self.cgan = self.build_cgan()
        self.cgan.compile(loss='binary_crossentropy', optimizer=Adam())

        # for saving metrics
        self.accuracies = []
        self.losses = []


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

    def build_cgan_generator(self):
        # random noise vector z
        z = Input(shape=(self.z_dim, ))

        # conditioning label: integer 0-9 specifying the number G should generate
        label = Input(shape=(1, ), dtype='int32')

        # label embedding:
        # ----------------
        # turns labels into dense vectors of size z_dim
        # produces 3D tensor with shape (batch_size, 1, z_dim)
        label_embedding = Embedding(self.num_classes, self.z_dim, input_length=1)(label)

        # flatten the embedding 3D tensor into 2D tensor with shape (batch_size, z_dim)
        label_embedding = Flatten()(label_embedding)

        # element-wise product of the vectors z and the label embeddings
        joined_representation = Multiply()([z, label_embedding])

        generator = self.build_generator()

        # generate image for the given label
        conditioned_img = generator(joined_representation)

        return Model([z, label], conditioned_img)

    def build_discriminator(self):
        model = Sequential()

        # convolutional layer, from 28x28x2 into 14x14x64 tensor
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(self.img_shape[0], self.img_shape[1], self.img_shape[2] + 1), padding='same'))
        model.add(LeakyReLU(alpha=0.01))

        # convolutional layer, from 14x14x64 into 7x7x64 tensor
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
        model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # output layer with sigmoid activation
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

    def build_cgan_discriminator(self):
        # input image
        img = Input(shape=self.img_shape)

        # label for the input image
        label = Input(shape=(1, ), dtype='int32')

        # label embedding:
        # ----------------
        # turns labels into dense vectors of size z_dim
        # produces 3D tensor with shape (batch_size, 1, 28*28*1)
        label_embedding = Embedding(self.num_classes, np.prod(self.img_shape), input_length=1)(label)

        # flatten the embedding 3D tensor into 2D tensor with shape (batch_size, 28*28*1)
        label_embedding = Flatten()(label_embedding)

        # reshape label embeddings to have same dimensions as input images
        label_embedding = Reshape(self.img_shape)(label_embedding)

        # concatenate images with their label embeddings
        concatenated = Concatenate(axis=-1)([img, label_embedding])

        discriminator = self.build_discriminator()

        # classify the image-label pair
        classification = discriminator(concatenated)

        return Model([img, label], classification)

    def build_cgan(self):
        # random noise vector z
        z = Input(shape=(self.z_dim, ))

        # image label
        label = Input(shape=(1, ))

        # generated image for that label
        img = self.generator([z, label])

        classification = self.discriminator([img, label])

        # combined Generator -> Discriminator model
        # G([z, lablel]) = x*
        # D(x*) = classification
        model = Model([z, label], classification)

        return model

    def train(self, iterations, batch_size, sample_interval):
        # load the MNIST dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # rescale [0, 255] grayscale pixel values to [-1, 1]
        X_train = X_train/127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # labels for real images: all ones
        real = np.ones((batch_size, 1))

        # labels for fake images: all zeros
        fake = np.zeros((batch_size, 1))

        for iteration in range(iterations):
            # -------------------------
            #  Train the Discriminator
            # -------------------------

            # get a random batch of real images and their labels
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict([z, labels])

            # train the Discriminator
            d_loss_real =  self.discriminator.train_on_batch([imgs, labels], real)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train the Generator
            # ---------------------

            # generate a batch of noise vectors
            z = np.random.normal(0, 1, (batch_size, self.z_dim))

            # get a batch of random labels
            labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # train the Generator
            g_loss = self.cgan.train_on_batch([z, labels], real)

            if (iteration + 1) % sample_interval == 0:
                # output training progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration + 1, d_loss[0], 100*d_loss[1], g_loss))

                # save losses and accuracies so they can be plotted after training
                self.losses.append((d_loss[0], g_loss))
                self.accuracies.append(100*d_loss[1])

                # output sample of generated images
                self.sample_images(iteration + 1)

    def sample_images(self, iteration, image_grid_rows=2, image_grid_columns=5):
        # sample random noise
        z = np.random.normal(0, 1, (image_grid_rows*image_grid_columns, self.z_dim))

        # get image labels 0-9
        labels = np.arange(0, 10).reshape(-1, 1)

        # generate images from random noise
        gen_imgs = self.generator.predict([z, labels])

        # rescale image pixel values to [0, 1]
        gen_imgs = 0.5*gen_imgs + 0.5

        # set image grid
        fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(10, 4), sharey=True, sharex=True)

        cnt = 0
        for i in range(image_grid_rows):
            for j in range(image_grid_columns):
                # output a grid of images
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title("Digit: %d" % labels[cnt])
                cnt += 1
        fig.savefig("out/images_%d.png" % iteration)
        plt.close()

    def generate_images(self):
        # set grid dimensions
        image_grid_rows = 10
        image_grid_columns = 5

        # sample random noise
        z = np.random.normal(0, 1, (image_grid_rows*image_grid_columns, self.z_dim))

        # get image labels to generate: 5 samples for each label
        labels_to_generate = np.array([[i for j in range(5)] for i in range(10)])
        labels_to_generate = labels_to_generate.flatten().reshape(-1, 1)

        # generate images from random noise
        gen_imgs = self.generator.predict([z, labels_to_generate])

        # rescale image pixel values to [0, 1]
        gen_imgs = 0.5*gen_imgs + 0.5

        # set image grid
        fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(10, 20), sharey=True, sharex=True)

        cnt = 0
        for i in range(image_grid_rows):
            for j in range(image_grid_columns):
                # output a grid of images
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title("Digit: %d" % labels_to_generate[cnt]) 
                cnt += 1
        fig.savefig("out/generated_images.png")
        plt.close()

def main():
    cgan = CGAN()
    cgan.train(iterations=12000, batch_size=32, sample_interval=1000)

    # now that the generator is trained, generate some targeted images
    cgan.generate_images()

if __name__ == "__main__":
    main()
