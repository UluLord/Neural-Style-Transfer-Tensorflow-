import tensorflow as tf

def style_transfer_layers(style_weight, content_weight):
    """
    This function returns the style and content layers used in the style transfer model.
    
    Args:
    - style_weight (float): weight for style loss in the style transfer algorithm.
    - content_weight (float): weight for content loss in the style transfer algorithm.
    
    Returns:
    - style_content_layers (list of tuples): a list of tuples including layer names and their weights for both style and content loss.
    """
    # List of 5 layers for style loss and their weights
    style_layers = [("block1_conv1", style_weight),
                    ("block2_conv1", style_weight),
                    ("block3_conv1", style_weight),
                    ("block4_conv1", style_weight),
                    ("block5_conv1", style_weight)]

    # Layer for content loss and its weight
    content_layer = [("block5_conv4", content_weight)]
    
    # List of all layers for both style and content loss
    style_content_layers = style_layers + content_layer

    return style_content_layers

def get_model(weights_dir, style_weight, content_weight, height, width):
    """
    This function loads the VGG19 model, sets its layers as untrainable, and gets outputs from layers specified in style_content_layers.
    
    Args:
    - weights_dir (str): path to the weights for the VGG19 model. If None, it will use the weights pretrained on the ImageNet dataset.
    - style_weight (float): weight for style loss in the style transfer algorithm.
    - content_weight (float): weight for content loss in the style transfer algorithm.
    - height (int): height of image.
    - width (int): width of image.

    Returns:
    - model_outputs (Tensorflow Model object): VGG19 model with specified outputs.
    - style_layers (list of tuples): a list of tuples including layer names and their weights for style loss.
    """
    # Check if weights_dir is specified or not, if not then load VGG19 with imagenet weights
    if weights_dir is None:
      model = tf.keras.applications.VGG19(input_shape=(height, width, 3),
                                          include_top=False,
                                          weights="imagenet")
    else:
      model = tf.keras.applications.VGG19(input_shape=(height, width, 3),
                                          include_top=False,
                                          weights=weights_dir)
    # Set all layers as untrainable  
    for layer in model.layers:
      layer.trainable = False 

    # Call layers
    style_content_layers = style_transfer_layers(style_weight, content_weight)
    
    # Get outputs from specified layers
    layer_outputs = [model.get_layer(layer[0]).output for layer in style_content_layers]
    
    # Create a new model with specified inputs and outputs
    model_outputs = tf.keras.Model(inputs=[model.input], outputs=layer_outputs)
    
    # Get style layers from style content layers
    style_layers = style_content_layers[:-1]
    
    return model_outputs, style_layers

def get_optimizer(model_optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum):
    """
    This function sets optimizer with it's parameters.
    
    Args:
    - model_optimizer (str): model optimizer type.
    - learning_rate (float): learning rate used during training.
    - beta_1 (float): the first hyperparameter for the Adam optimizer.
    - beta_2 (float): the second hyperparameter for the Adam optimizer.
    - epsilon (float): a small constant added to the denominator to prevent division by zero for the Adam optimizer.
    - momentum (float): momentum term for the SGD optimizer.
    - nesterov (bool): whether to use Nesterov momentum for the SGD optimizer.
    - rho (float): decay rate for the moving average of the squared gradient for the RMSprop optimizer. 
    - rmsprop_momentum (float): momentum term for the RMSprop optimizer.
    
    Returns:
    - optimizer: model optimizer with it's parameters.
    
    Raises:
    - ValueError: if the optimizer is not recognized.
    """
    # Initialize optimizer variable
    optimizer = None

    # Check if the model optimizer is SGD
    if model_optimizer == "sgd":
        # Use the SGD optimizer with specified learning rate, momentum, and nesterov attributes
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=momentum,
                                            nesterov=nesterov)
    # Check if the model optimizer is RMSprop
    elif model_optimizer == "rmsprop":
        # Use the RMSprop optimizer with specified learning rate, rho, and momentum attributes
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                                rho=rho,
                                                momentum=rmsprop_momentum)
    # Check if the model optimizer is Adam
    elif model_optimizer == "adam":
        # Use the Adam optimizer with specified learning rate, beta_1, beta_2, and epsilon attributes
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                            beta_1=beta_1,
                                            beta_2=beta_2,
                                            epsilon=epsilon)
        
    # Give error if unsupported optimizer type specified.
    else:
      raise ValueError("Unsupported optimizer format: {}".format(model_optimizer))

    return optimizer