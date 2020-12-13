import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from IPython import display
from skimage import transform

class Interpolate:

    def __init__(self):
        # We could retrieve this value from module.get_input_shapes() if we didn't know
        # beforehand which module we will be using.
        self.latent_dim = 512

    # Interpolates between two vectors that are non-zero and don't both lie on a
    # line going through origin. First normalizes v2 to have the same norm as v1. 
    # Then interpolates between the two vectors on the hypersphere.
    def interpolate_hypersphere(self, v1, v2, num_steps):
        v1_norm = tf.norm(v1)
        v2_norm = tf.norm(v2)
        v2_normalized = v2 * (v1_norm / v2_norm)

        vectors = []
        for step in range(num_steps):
            interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
            interpolated_norm = tf.norm(interpolated)
            interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
            vectors.append(interpolated_normalized)

        return tf.stack(vectors)

    # Given a set of images, save an animation.
    def animate(self, images):
        converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
        imageio.mimsave('out/animation.gif', converted_images)

    def interpolate_between_vectors(self):
        with tf.Graph().as_default():
            module = hub.Module("https://tfhub.dev/google/progan-128/1")

            # Change the seed to get different random vectors.
            v1 = tf.compat.v1.random_normal([self.latent_dim], seed=3)
            v2 = tf.compat.v1.random_normal([self.latent_dim], seed=1)
    
            # Creates a tensor with 50 steps of interpolation between v1 and v2.
            vectors = self.interpolate_hypersphere(v1, v2, 25)

            # Uses module to generate images from the latent space.
            interpolated_images = module(vectors)

            with tf.compat.v1.Session() as session:
                session.run(tf.compat.v1.global_variables_initializer())
                interpolated_images_out = session.run(interpolated_images)

            self.animate(interpolated_images_out)

def main():
    interpolater = Interpolate()
    interpolater.interpolate_between_vectors()

if __name__ == "__main__":
    main()

