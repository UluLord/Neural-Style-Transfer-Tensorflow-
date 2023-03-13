import numpy as np
import tensorflow as tf

class ComputeLoss:
    """
    This class calculates the losses of the style transfer algorithm.
    """
    def __init__(self):
        """
        Initialize the loss computation class.
        """
    def content(self, content_image_output, generated_image_output):
        """
        This function computes the content loss between the content image and the generated image.

        Args:
        - content_image_output (tensor): output tensor of the content image.
        - generated_image_output (tensor): output tensor of generated image.

        Returns:
        - content_loss_normalized (tensor): normalized content loss.
        """
        # Get the content image and generated image output from the final layer
        a_C = content_image_output[-1]
        a_G = generated_image_output[-1]

        # Get the dimensions of the generated image
        m, h, w, c = a_G.get_shape().as_list()

        # Flatten the content image and generated image
        a_C_unrolled = tf.reshape(a_C, [m, h*w, c])
        a_G_unrolled = tf.reshape(a_G, [m, h*w, c])

        # Calculate the content loss normalized by dividing by 4 times the product of height, width and channels
        content_loss = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
        normalizer = 4*h*w*c
        content_loss_normalized = content_loss / normalizer

        return content_loss_normalized
  
    def style(self, style_image_output, generated_image_output, style_layers):
        """
        This function computes the total style loss for the generated image.

        Args:
        - style_image_output (tensor): output tensor of the style image.
        - generated_image_output (tensor): output tensor of generated image.
        - style_layers (list of tuples): a list of tuples including layer names and their weights for style loss.

        Returns:
        - style_loss (tensor): normalized style loss.
        """
        # Initialize style loss
        style_loss = 0

        # Get the activation tensors for the style image and the generated image for each layer
        a_S = style_image_output[:-1]
        a_G = generated_image_output[:-1]

        # Loop over each layer and compute the style loss for each layer
        for i, weight in zip(range(len(a_G)), style_layers):
          _, h, w, c = a_G[i].get_shape().as_list()

          # Flatten the activations into 2D tensors
          a_S_flatten = tf.transpose(tf.reshape(a_S[i], [-1, c]))
          a_G_flatten = tf.transpose(tf.reshape(a_G[i], [-1, c]))
  
          # Compute gram matrices of the activations
          GS = tf.matmul(a_S_flatten, tf.transpose(a_S_flatten))
          GG = tf.matmul(a_G_flatten, tf.transpose(a_G_flatten))
  
          # Compute the style loss as the squared difference between the Gram matrices, and 
          # normalize the style loss by dividing by 4 times the square of the number of activations elements
          style_layer_loss = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
          normalizer = 4*np.power(h*w*c, 2)
          style_layer_loss_normalized = style_layer_loss / normalizer

          # Add the weighted style loss for the current layer to the total style cost
          style_loss += weight[1] * style_layer_loss_normalized

        return style_loss

    def total(self, content_loss, style_loss, alpha, beta):
        """
        This function computes the total loss for generated image.

        Args:
        - content_loss (tensor): content loss for generated image.
        - style_loss (tensor): style loss for generated image.
        - alpha (int): weight to apply on the content loss.
        - beta (int): weight to apply on the style loss.

        Returns:
        - total_loss (tensor): total loss for the generated image.
        """
        # Compute the total loss by combining the content loss and the style loss
        total_loss = tf.add(content_loss*alpha, style_loss*beta)

        return total_loss