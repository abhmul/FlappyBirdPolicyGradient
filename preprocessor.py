import numpy as np

from keras.preprocessing.image import img_to_array

def preprocess_image(image):
    """
    Preprocesses an image into a binary image
    :param image: image to process
    :return: binary version of the image as an array
    """

    img_array = img_to_array(image.convert('L'))
    return np.around(img_array / 255.0)

# def combine_screen(images, )


