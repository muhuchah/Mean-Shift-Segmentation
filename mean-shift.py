import numpy as np
import matplotlib.pyplot as plt
from skimage import io


from utils import save_image 


if __name__ == "__main__":
    image = io.imread('input.jpg')
    print("Image shape: ", image.shape)

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    save_image(image, title="Original Image")
