import numpy as np
from PIL.Image import fromarray
from PIL import Image
try:
    from pygame.surfarray import array3d
except ImportError:
    array3d = None

from keras.preprocessing.image import img_to_array, array_to_img

RESIZE = (80, 80)

def preprocess_image(image):
    """
    Preprocesses an image into a binary image
    :param image: image to process
    :return: binary version of the image as an array
    """
    # image.convert('L').save('test.png')
    img_array = img_to_array(image)[0]
    # print img_array.shape
    img_array /= 255.0
    img_array = img_array > 0.0
    return img_array

def pyg_to_pil(image):
    arr_img = array3d(image)
    return fromarray(arr_img).convert('L')

def combine_screen(images, screen_size):
    """
    Merge sprite images into desired screen image
    :param images: A list of tuples with first index image, and second index position in screen
    :param screen_size: a tuple with the size of the display screen
    :return: The combined image with only the selected portions white
    """

    # Create the background screen
    image = Image.new('L', screen_size)
    # print image.size
    for img, (posx, posy) in images:
        # Overlay the images
        image.paste(pyg_to_pil(img), (int(posy), int(posx)))
    # image.save('test.png')
    # Turn it into an array and return
    return image

def preprocess_screen(images, screen_size):

    screen = combine_screen(images, screen_size)
    return preprocess_image(screen.resize(RESIZE))

def clean_screen(arr_img):
    return array_to_img((arr_img*255).reshape((1,)+(RESIZE[1], RESIZE[0])))