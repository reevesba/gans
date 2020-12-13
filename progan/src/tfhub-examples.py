import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

def tfhub_progan():
    with tf.Graph().as_default():
        # import the progressive GAN from TFHub
        module = hub.Module("https://tfhub.dev/google/progan-128/1")

        # latent dimension that gets 
        latent_dim = 512

        # change the seed to get different faces.
        latent_vector = tf.random.normal([1, latent_dim], seed=1337)

        # uses module to generate images from the latent space.
        interpolated_images = module(latent_vector)

        # runs the tensorflow session and gets back the image in shape (1, 128, 128, 3)
        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            image_out = session.run(interpolated_images)

    plt.imshow(image_out.reshape(128, 128, 3))
    plt.savefig("out/tf_example_1.png")

def main():
    tfhub_progan()

if __name__=="__main__":
    main()


