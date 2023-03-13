import numpy as np
import tensorflow as tf

def load_image(image_path, height, width):
    """
    This function loads an image from path.

    Args:
    - image_path (str): path to an image.
    - height (int): height of image.
    - width (int): width of image.

    Returns:
    - image (tensor): an image in tensor format.
    """
    # Load image and resize it
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(height, width))

    # Convert to numpy array with type uint8
    image = np.uint8(image)

    # Add an extra dimension
    image = np.expand_dims(image, axis=0)

    # Set as constant tensor
    image = tf.constant(image)

    return image

def add_noise(image):
    """
    This function adds noise to an image, and clips it.

    Args:
    - image (tensor): an image to make noisy.

    Returns:
    - generated_image (tensor): the generated image that is noisy and clipped.
    """
    tf.random.set_seed(1234)

    # Convert the content image to a tensor with type float32
    generated_image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Add random normal noise to the generated image
    noise = tf.random.normal(tf.shape(generated_image), mean=0., stddev=0.2)

    # Add noise to content image to generate the noise image
    generated_image = tf.add(generated_image, noise)

    # Clip the values of the generated image between 0 and 1
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0., clip_value_max=1.)

    # Convert the tensor to a variable
    generated_image = tf.Variable(generated_image)

    return generated_image
  
def image_encoding(model, image):
    """
    This function encodes an image into feature representations.

    Args:
    - model (Tensorflow Model object): a model with specified inputs and outputs.
    - image (tensor): an image in tensor format.

    Returns:
    - feature_rep (tensor): feature representations of the image.
    """
    # Convert the image to a tensor of type float32 and convert it to a variable
    preprocessed_image = tf.Variable(tf.image.convert_image_dtype(image, dtype=tf.float32))
    
    # Pass the preprocessed image through the VGG19 model to obtain its feature representations
    feature_rep = model(preprocessed_image)

    return feature_rep