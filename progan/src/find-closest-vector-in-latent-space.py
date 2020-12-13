import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

class ClosestVector:

    def __init__(self):
        self.latent_dim = 512
        self.image_from_module_space = True

        if self.image_from_module_space:
            self.target_image = self.get_module_space_image()
        else:
            self.target_image = upload_image()

    # Simple way to display an image.
    def display_image(self):
        plt.figure()
        plt.axis("off")
        plt.imshow(self.target_image)
        plt.savefig("out/sample_image")

    # Display multiple images in the same figure.
    def display_images(self, images, captions=None):
        num_horizontally = 5
        f, axes = plt.subplots(len(images) // num_horizontally, num_horizontally, figsize=(20, 20))

        for i in range(len(images)):
            axes[i // num_horizontally, i % num_horizontally].axis("off")
            if captions is not None:
                axes[i // num_horizontally, i % num_horizontally].text(0, -3, captions[i])
            axes[i // num_horizontally, i % num_horizontally].imshow(images[i])
        f.tight_layout()
        f.savefig("out/sample_grid")

    def get_module_space_image(self):
        with tf.Graph().as_default():
            module = hub.Module("https://tfhub.dev/google/progan-128/1")
            vector = tf.compat.v1.random_normal([1, self.latent_dim], seed=4)
            images = module(vector)

            with tf.compat.v1.Session() as session:
                session.run(tf.compat.v1.global_variables_initializer())
                image_out = session.run(images)[0]
        return image_out

    def upload_image(self):
        uploaded = files.upload()
        image = imageio.imread(uploaded[uploaded.keys()[0]])
        return transform.resize(image, [128, 128])

    def find_closest_latent_vector(self, num_optimization_steps):
        images = []
        losses = []

        with tf.Graph().as_default():
            module = hub.Module("https://tfhub.dev/google/progan-128/1")

            initial_vector = tf.compat.v1.random_normal([1, self.latent_dim], seed=5)

            vector = tf.compat.v1.get_variable("vector", initializer=initial_vector)
            image = module(vector)

            target_image_difference = tf.reduce_sum(tf.compat.v1.losses.absolute_difference(image[0], self.target_image[:, :, :3]))

            # The latent vectors were sampled from a normal distribution. We can get
            # more realistic images if we regularize the length of the latent vector to 
            # the average length of vector from this distribution.
            regularizer = tf.abs(tf.norm(vector) - np.sqrt(self.latent_dim))
    
            loss = target_image_difference + regularizer
    
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.3)
            train = optimizer.minimize(loss)

            with tf.compat.v1.Session() as session:
                session.run(tf.compat.v1.global_variables_initializer())
                for _ in range(num_optimization_steps):
                    _, loss_out, im_out = session.run([train, loss, image])
                    images.append(im_out[0])
                    losses.append(loss_out)
                    print(loss_out)
            return images, losses

def main():
    cv = ClosestVector()
    result = cv.find_closest_latent_vector(num_optimization_steps=40)
    captions = [ f'Loss: {l:.2}' for l in result[1]]

    cv.display_image()
    cv.display_images(result[0], captions)

if __name__ == "__main__":
    main()



