import tensorflow as tf
import keras as K

'''
The following methods implement some of the key innovations in GAN training:
    1. progressively growing and smoothly fading in high-res layers
    2. mini-batch standard deviation
    3. equalized learning rate
    4. pixel-wise feature normalization
'''

# progressively growing and smoothly fading in high-res layers
def upscale_layer(layer, upscale_factor):
    '''
    Upscales layer (tensor) by the factor (int) where
    the tensor is [group, height, width, channels]
    '''
    height, width = layer.get_shape()[1:3]
    size = (upscale_factor*height, upscale_factor*width)
    upscaled_layer = tf.image.resize_nearest_neighbor(layer, size)

    return upscaled_layer

def smoothly_merge_last_layer(list_of_layers, alpha):
    '''
    Smoothly merges in a layer based on a threshold value alpha.
    This function assumes: that all layers are already in RGB. 
    This is the function for the Generator.
    :list_of_layers	:	items should be tensors ordered by size
    :alpha 			: 	float \in (0,1)
    '''
    # if you are using pure Tensorflow rather than Keras, always remember scope
    last_fully_trained_layer = list_of_layers[-2]

    # now we have the originally trained layer
    last_layer_upscaled = upscale_layer(last_fully_trained_layer, 2)

    # this is the newly added layer not yet fully trained
    larger_native_layer = list_of_layers[-1]

    # this makes sure we can run the merging code
    assert larger_native_layer.get_shape() == last_layer_upscaled.get_shape()

    # this code block should take advantage of broadcasting
    new_layer = (1 - alpha)*upscaled_layer + larger_native_layer*alpha

    return new_layer

# mini-batch standard deviation
def minibatch_std_layer(layer, group_size=4):
    '''
    Will calculate minibatch standard deviation for a layer.
    Will do so under a pre-specified tf-scope with Keras.
    Assumes layer is a float32 data type. Else needs validation/casting.
    NOTE: there is a more efficient way to do this in Keras, but just for
    clarity and alignment with major implementations (for understanding) 
    this was done more explicitly. Try this as an exercise.
    '''
    # Hint!
    # if you are using pure Tensorflow rather than Keras, always remember scope
    # minibatch group must be divisible by (or <=) group_size
    group_size = K.backend.minimum(group_size, tf.shape(layer)[0])

    # just getting some shape information so that we can use
    # them as shorthand as well as to ensure defaults
    shape = list(K.int_shape(input))
    shape[0] = tf.shape(input)[0]

    # reshaping so that we operate on the level of the minibatch
    # in this code we assume the layer to be:
    # [Group (G), Minibatch (M), Width (W), Height (H) , Channel (C)]
    # but be careful different implementations use the Theano specific
    # order instead
    minibatch = K.backend.reshape(layer, (group_size, -1, shape[1], shape[2], shape[3]))

    # center the mean over the group [M, W, H, C]
    minibatch -= tf.reduce_mean(minibatch, axis=0, keepdims=True)

    # calculate the variance of the group [M, W, H, C]
    minibatch = tf.reduce_mean(K.backend.square(minibatch), axis = 0)

    # calculate the standard deviation over the group [M, W, H, C]
    minibatch = K.backend.square(minibatch + 1e8)

    # take average over feature maps and pixels [M, 1, 1, 1]
    minibatch = tf.reduce_mean(minibatch, axis=[1,2,4], keepdims=True)

    # add as a layer for each group and pixels
    minibatch = K.backend.tile(minibatch, [group_size, 1, shape[2], shape[3]])

    # append as a new feature map
    return K.backend.concatenate([layer, minibatch], axis=1)

# equalized learning rate
def equalize_learning_rate(shape, gain, fan_in=None):
    '''
    This adjusts the weights of every layer by the constant from
    He's initializer so that we adjust for the variance in the dynamic range
    in different features
    shape   :   shape of tensor (layer): these are the dimensions of each layer.
    For example, [4,4,48,3]. In this case, 
        [kernel_size, kernel_size, number_of_filters, feature_maps]. 
        But this will depend slightly on your implementation.
    gain    :   typically sqrt(2)
    fan_in  :   adjustment for the number of incoming connections as per Xavier's / He's initialization 
    '''
    # default value is product of all the shape dimension minus the feature maps dim -- this gives us the number of incoming connections per neuron
    if fan_in is None: fan_in = np.prod(shape[:-1])
    
    # this uses He's initialization constant (He et al, 2015)
    std = gain/K.sqrt(fan_in)

    # creates a constant out of the adjustment
    wscale = K.constant(std, name='wscale', dtype=np.float32)

    # gets values for weights and then uses broadcasting to apply the adjustment
    adjusted_weights = K.get_value('layer', shape=shape, initializer=tf.initializers.random_normal())*wscale

    return adjusted_weights

# pixel-wise feature normalization
def pixelwise_feat_norm(inputs, **kwargs):
	'''
	Uses pixelwise feature normalization as proposed by
	Krizhevsky et at. 2012. Returns the input normalized
	:inputs 	: 	Keras / TF Layers 
	'''
	normalization_constant = K.backend.sqrt(K.backend.mean(inputs**2, axis=-1, keepdims=True) + 1.0e-8)
	return inputs/normalization_constant








