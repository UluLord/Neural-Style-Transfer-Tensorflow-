# **Neural Style Transfer [Tensorflow]**

This repository contains an implementation of neural style transfer. Neural Style Transfer (NST) is a deep learning technique that allows us to apply the artistic style of one image onto the content of another image. It is achieved by optimizing a loss function that compares the feature representations of the style and content images.

This technique is based on Convolutional Neural Networks (CNNs), specifically the VGG network, which is pre-trained on millions of images. The VGG network is used as a feature extractor to capture both the content and style information from the input images.

To perform the NST, we need to define a content image and a style image, and then create a new image that has the content of the former and the style of the latter. The process involves iteratively adjusting the pixels of the generated image to minimize the style and content loss.

![NST Model](https://user-images.githubusercontent.com/99184963/224574758-85e3ff1e-2673-4b1e-8f02-e12b94427786.png)
>Retrieved from [this link](https://towardsdatascience.com/how-do-neural-style-transfers-work-b76de101eb3) 

## **1. Usage**

### *1.1. Cloning*
To use the NST in your images, clone this repository using your terminal like following command;

    git clone https://github.com/UluLord/Neural-Style-Transfer-Tensorflow-.git

After cloning, change the directory, you are working, to this repository directory;

    cd Neural-Style-Transfer-Tensorflow-

### *1.2 Requirements*

This work has been tested on these libraries;

* Tensorflow: 2.11.0
* Numpy: 1.22.4
* Matplotlib: 3.7.0

To install the required packages, run the following command;

    pip install -r requirements.txt

**NOTE:** It may work with other versions of the libraries, but this has not been tested.

* This work has also been tested on NVIDIA GeForce RTX 3060 GPU.

**NOTE:** It is highly recommended to work with a GPU.

### *1.3. Generate a Styled Image*

Then, use the **main.py** with desired parameters to generate a styled image

***Parameters***
  * **content_image_path:** Path to a content image. Required.
  * **style_image_path:** Path to a style image. Required.
  * **height:** Height of image. Default is 500.
  * **width:** Width of image. Default is 500.
  * **epochs:** Number of training epochs. Default is 10000.
  * **weights_dir:** Path to the weights for the VGG19 model. If None, the model will use the weights, pretrained on the ImageNet dataset. Default is None.
  * **content_weight:** Weight for content loss in the style transfer algorithm. Default is 1.
  * **style_weight:** Weight for style loss in the style transfer algorithm. Default is 0.1.
  * **alpha:** Weight to apply on the content loss. Default is 10.
  * **beta:** Weight to apply on the style loss. Default is 40.
  * **optimizer:** Model optimizer type (choose one of those: sgd, rmsprop or adam). Default is 'adam'.
  * **learning_rate:** Learning rate used during training. Default is 0.01.
  * **beta_1:** The first hyperparameter for the Adam optimizer. Default is 0.9.
  * **beta_2:** The second hyperparameter for the Adam optimizer. Default is 0.999.
  * **epsilon:** A small constant added to the denominator to prevent division by zero for the Adam optimizer. Default is 1e-7.
  * **momentum:** Momentum term for the SGD optimizer. Default is 0.
  * **nesterov:** Whether to use Nesterov momentum for the SGD optimizer. Default is False.
  * **rho:** Decay rate for the moving average of the squared gradient for the RMSprop optimizer. Default is 0.9.
  * **rmsprop_momentum:** Momentum term for the RMSprop optimizer. Default is 0.
  * **patience:** Number of epochs to wait before showing generated image. Default is 500.

***Example Usage***

    python main.py --content_image_path ./images/Istanbul.jpg --style_image_path ./images/Starry_Night.jpg --height 500 --width 500 --epochs 20000 --content_weight 1. --style_weight 0.1 --alpha 10 --beta 40 --optimizer adam --patience 500

***Some of Generated Images***

1. Using an Istanbul image as content image, and a picture named 'The Starry Night' drawn by Vincent van Gogh as style image:

  * A gif showing the training process;

    <img src="https://user-images.githubusercontent.com/99184963/224575241-a1f3f63c-70eb-44c1-b240-445cc0aef741.gif" width="300" height="300">

  * A generated image;

    <img src="https://user-images.githubusercontent.com/99184963/224574833-61f16ac4-fdfe-4497-bfc4-296269c5aa94.jpg" width="300" height="300">

  * A figure comparing content image, style image, and generated image;

    <img src="https://user-images.githubusercontent.com/99184963/224574907-23e1faf6-8d80-48aa-9fac-142cfb24efc2.jpg" width="900" height="300">

2. Using an image showing me as content image :), and a picture showing Vincent van Gogh with colored type as style image:

  * A gif showing the training process;

    <img src="https://user-images.githubusercontent.com/99184963/224575376-5f333ee1-928c-44d1-8526-f92b4100c614.gif" width="300" height="300">

  * A generated image;

    <img src="https://user-images.githubusercontent.com/99184963/224575042-bebc0373-4302-4b13-84da-2d582bb3c52a.jpg" width="300" height="300">

  * A figure comparing content image, style image, and generated image;

    <img src="https://user-images.githubusercontent.com/99184963/224575058-c9b29b63-6de6-42e6-81fd-9810012ef160.jpg" width="900" height="300">


**NOTE:** It is possible to get better results by adjusting the parameters.

## **2. Citation**

If you use this repository in your work, please consider citing us as the following.

    @misc{ululord2023neural-style-transfer-tensorflow,
	      author = {Fatih Demir},
          title = {Neural Style Transfer [Tensorflow]},
          date = {2023-03-13},
          url = {https://github.com/UluLord/Neural-Style-Transfer-Tensorflow-}
          }
