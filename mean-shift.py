import numpy as np
import matplotlib.pyplot as plt
from skimage import io


from utils import save_image 


def feature_space(image):
    rows, cols, channels = image.shape

    # Reshape the image into a 2D array (pixels x features)
    # We include spatial coordinates (x, y) as additional features
    x_coords = np.arange(rows)[:, np.newaxis]  # Row indices
    y_coords = np.arange(cols)[:, np.newaxis]  # Column indices

    # Combine color and spatial features
    X_color = image.reshape(-1, channels)  # Flatten color channels
    X_spatial = np.hstack([np.repeat(x_coords, cols, axis=0), np.tile(y_coords, (rows, 1))])  # Spatial coordinates

    # Normalize spatial coordinates to [0, 1]
    X_spatial = X_spatial / np.max(X_spatial)

    # Combine color and spatial features
    X = np.hstack([X_color, X_spatial])
    print("Feature matrix shape:", X.shape)

    return X


if __name__ == "__main__":
    image = io.imread('input.jpg')
    print("Image shape: ", image.shape)

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    save_image(image, title="Original Image")

    X = feature_space(image)

