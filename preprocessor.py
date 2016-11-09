import numpy as np
from PIL import Image

from keras.preprocessing.image import img_to_array, array_to_img

RESIZE = (94, 94)

def preprocess_image(image):
    """
    Preprocesses an image into a binary image
    :param image: image to process
    :return: binary version of the image as an array
    """

    img_array = img_to_array(image.convert('L'))
    return np.around(img_array / 255.0)

def combine_screen(images, screen_size):
    """
    Merge sprite images into desired screen image
    :param images: A list of tuples with first index image, and second index position in screen
    :param screen_size: a tuple with the size of the display screen
    :return: The combined image with only the selected portions white
    """

    # Create the image array
    screen = np.zeros(screen_size)
    for img, (posx, posy) in images:
        img_arr = preprocess_image(img)
        # Get the right and top edge values
        endx = min(screen_size[0], posx + img_arr.shape[0])
        endy = min(screen_size[1], posy + img_arr.shape[1])
        # Put the img onto the screen
        screen[posx:endx, posy:endy] = img_arr
    return screen

def preprocess_screen(images, screen_size):

    screen = combine_screen(images, screen_size)
    return preprocess_image(array_to_img(screen).resize(RESIZE))