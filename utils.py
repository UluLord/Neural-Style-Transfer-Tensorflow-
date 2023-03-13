import os
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_array(tensor):
    """
    This function converts a tensor to a numpy array with type uint8.
    
    Args:
    - tensor (tensor): tensor to be converted to a numpy array.
    
    Returns:
    - arrayed_tensor (array): numpy array with type uint8.
    """
    # Multiply the tensor with 255 to scale it from 0 to 255
    tensor = tensor*255

    # Convert the tensor to numpy array with dtype "uint8"
    arrayed_tensor = np.array(tensor, dtype="uint8")
    
    # Check if the number of dimensions is greater than 3
    if np.ndim(arrayed_tensor)>3:
        # Assert that the first dimension of the arrayed_tensor is equal to 1
        assert arrayed_tensor.shape[0] == 1, f"Wrong shape: expected arrayed_tensor.shape[0] is equal to 1, but got {arrayed_tensor.shape[0]}"
        # Get the first element of the arrayed_tensor, because it only has one channel
        arrayed_tensor = arrayed_tensor[0]
    
    return arrayed_tensor

def show_epoch_result(generated_image, cur_epoch, last_epoch):
    """
    This function plots the generated image of the given epoch and saves it if it is the last epoch.
    
    Args:
    - generated_image (tensor): style transferred image.
    - cur_epoch (int): current epoch.
    - last_epoch (int): last epoch.
    
    Returns:
    - None
    """
    # Specify directory to save the generated images
    save_to_dir = "./generated_images"
    
    # Convert the generated image type from tensor to array
    arrayed_tensor = tensor_to_array(generated_image) 
    
    # Plot and show the image
    plt.figure(figsize=(16,12))
    plt.imshow(arrayed_tensor)
    plt.axis("off")
    
    # Save the image if given epoch is the last epoch
    if cur_epoch == last_epoch:
        # Check if the directory to save the image exists, otherwise create it
        if os.path.exists(save_to_dir) == False: 
            os.makedirs(save_to_dir)
        # Save the image
        plt.savefig(os.path.join(save_to_dir, f"generated_image.jpg"))
        
    # Show the image with title named current epoch 
    plt.title(f"Epoch {cur_epoch}")
    plt.show()

    
def compare_result(image_list):
    """
    This function plots a list of images and saves them if specified.

    Args:
    - image_list (list): a list of image tensors.

    Returns:
    - None
    """
    # Specify directory to save the plot
    save_dir = "./generated_images"
    
    # Specify titles to show in subplots
    titles = ["Content", "Style", "Generated"]

    plt.figure(figsize=(16, 12))
    for i in range(len(image_list)):
        # Plot each image in separate subplots
        plt.subplot(1, len(image_list), i+1) 
        plt.imshow(image_list[i][0])
        plt.axis("off")
        plt.title(titles[i])
    
    # Create the directory if it does not exist
    if os.path.exists(save_dir) == False: 
        os.makedirs(save_dir)
    # Save the plot
    plt.savefig(os.path.join(save_dir, "result.jpg"))
    # Show the plot
    plt.show()
    # Print a message indicating the saved path of the image
    print("The plot is saved at './generated_images/result.jpg'.")
