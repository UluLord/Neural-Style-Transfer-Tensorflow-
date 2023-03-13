import argparse
import tensorflow as tf
from preprocessing import load_image, add_noise, image_encoding
from losses import ComputeLoss
from utils import show_epoch_result, compare_result
from model import get_model, get_optimizer

def set_arguments():
    """
    This function parses command line arguments and returns them as a dictionary.
    """ 
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(
        # Description of the project
        description="This project implements a style transfer algorithm for generating new images by transferring the style of one image to the content of another image. \n\nTo generate new image, adjust the parameters if necessary:",
        # Usage string to display
        usage="Generating new images by transfering style",
        # Set the formatter class to ArgumentDefaultsHelpFormatter
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # Set prefix chars
        prefix_chars="-",
        # Set default value for argument_default
        argument_default=argparse.SUPPRESS,
        # Allow abbreviations of options
        allow_abbrev=True,
        # Add help argument
        add_help=True)

    # Add arguments
    parser.add_argument("--content_image_path", type=str, required=True, 
                        help="Path to a content image")
    parser.add_argument("--style_image_path", type=str, required=True, 
                        help="Path to a style image")
    parser.add_argument("--height", default=500, type=int, required=False, 
                        help="Height of image")
    parser.add_argument("--width", default=500, type=int, required=False, 
                        help="Width of image")
    parser.add_argument("--epochs", default=10000, type=int, required=False, 
                        help="Number of training epochs")
    parser.add_argument("--weights_dir", default=None, type=str, required=False, 
                        help="Path to the weights for the VGG19 model. If None, the model will use the weights, pretrained on the ImageNet dataset.")
    parser.add_argument("--content_weight", default=1., type=float, required=False,
                        help="Weight for content loss in the style transfer algorithm")
    parser.add_argument("--style_weight", default=0.1, type=float, required=False,
                        help="Weight for style loss in the style transfer algorithm")
    parser.add_argument("--alpha", default=10, type=int, required=False,
                        help="Weight to apply on the content loss")
    parser.add_argument("--beta", default=40, type=int, required=False,
                        help="Weight to apply on the style loss")
    parser.add_argument("--optimizer", default="adam", type=str, required=False, choices=["sgd", "rmsprop", "adam"], 
                        help="Model optimizer type (choose one of those: sgd, rmsprop or adam)")
    parser.add_argument("--learning_rate", default=0.01, type=float, required=False,
                        help="Learning rate used during training")
    parser.add_argument("--beta_1", default=0.9, type=float, required=False,
                        help="The first hyperparameter for the Adam optimizer")
    parser.add_argument("--beta_2", default=0.999, type=float, required=False, 
                        help="The second hyperparameter for the Adam optimizer")
    parser.add_argument("--epsilon", default=1e-7, type=float, required=False, 
                        help="A small constant added to the denominator to prevent division by zero for the Adam optimizer")
    parser.add_argument("--momentum", default=0., type=float, required=False, 
                        help="Momentum term for the SGD optimizer")
    parser.add_argument("--nesterov", default=False, type=bool, required=False, 
                        help="Whether to use Nesterov momentum for the SGD optimizer")
    parser.add_argument("--rho", default=0.9, type=float, required=False, 
                        help="Decay rate for the moving average of the squared gradient for the RMSprop optimizer") 
    parser.add_argument("--rmsprop_momentum", default=0., type=float, required=False, 
                        help="Momentum term for the RMSprop optimizer")
    parser.add_argument("--patience", default=500, type=int, required=False, 
                        help="Number of epochs to wait before showing generated image")  

    # Parse the arguments and convert them to a dictionary
    args = vars(parser.parse_args())

    return args


def training(content_image_output, style_image_output, generated_image, 
             model, optimizer, epochs, style_layers, alpha, beta, patience):
    """
    This function trains the style transfer model for a given number of epochs

    Args:
    - content_image_output (tensor): output tensor of the content image. 
    - style_image_output (tensor): output tensor of the style image.
    - generated_image (tensor): variable tensor of the generated image to be optimized.
    - model (Tensorflow Model object): style transfer model.
    - epochs (int): number of training epochs.
    - optimizer (optimizer): the optimizer to be used in training.
    - style_layers (list of tuples): a list of tuples including layer names and their weights for style loss.
    - alpha (float): weight to apply on the content loss.
    - beta (float): weight to apply on the style loss.
    - patience (int): number of epochs to wait before showing generated image.

    Returns:
    - None
    """
    # Get class including loss functions
    compute_loss = ComputeLoss()
    # Print a message indicating the start of model training
    print("Model training...")
    
    # Loop over each epoch
    for epoch in range(0, epochs+1):
      # Use gradient tape to keep track of gradients
      with tf.GradientTape() as tape:
          # Compute the output of the generated image
          generated_image_output = model(generated_image)
          # Compute the content loss
          content_loss = compute_loss.content(content_image_output, generated_image_output)
          # Compute the style loss
          style_loss = compute_loss.style(style_image_output, generated_image_output, style_layers)
          # Compute total loss
          total_loss  = compute_loss.total(content_loss, style_loss, alpha, beta)
      
      # Compute gradients with respect to the generated image
      grads = tape.gradient(total_loss, generated_image)
      # Apply the computed gradients to the generated image
      optimizer.apply_gradients([(grads, generated_image)])
      # Clip the pixel values of the generated image
      clipped_image = tf.clip_by_value(generated_image, clip_value_min=0., clip_value_max=1.)
      # Assign the clipped image to the generated image
      generated_image.assign(clipped_image)

      if epoch % patience == 0:
         # Print the training logs
         print(f"Epoch: {epoch} | Total Loss: {total_loss} | Content Loss: {content_loss} | Style Loss: {style_loss}")
         # Plots generated image in given epoch and saves the in last epoch.
         show_epoch_result(generated_image, epoch, epochs)

    # Print a message indicating the end of model training
    print("The training process has finished...")
    # Show the content image, style image, and generated image in a subplot to compare the result
    compare_result([content_image, style_image, generated_image])
    

if __name__ == "__main__":
    
   # Set arguments
   args = set_arguments()
    
   # Load images
   content_image = load_image(args["content_image_path"], args["height"], args["width"])
   style_image = load_image(args["style_image_path"], args["height"], args["width"])
   generated_image = add_noise(content_image)
    
   # Load model
   vgg_model_outputs, style_layers = get_model(args["weights_dir"], args["style_weight"], args["content_weight"], 
                                               args["height"], args["width"])
    
   # Encode images
   content_image_output = image_encoding(vgg_model_outputs, content_image)
   style_image_output = image_encoding(vgg_model_outputs, style_image)

   # Get optimizer
   optimizer = get_optimizer(args["optimizer"], args["learning_rate"], args["beta_1"], args["beta_2"], args["epsilon"], 
                              args["momentum"], args["nesterov"], args["rho"], args["rmsprop_momentum"])

   # Training process
   training(content_image_output, style_image_output, generated_image, 
            vgg_model_outputs, optimizer, args["epochs"], 
            style_layers, args["alpha"], args["beta"], args["patience"])