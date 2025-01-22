import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min


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


def mean_shift(X, bandwidth=0.1, max_iter=10, tol=1e-3):
    n_samples, n_features = X.shape
    centroids = X.copy()  # Initialize centroids as the input data

    # Precompute the pairwise distance matrix to save computation time
    nn = NearestNeighbors(radius=bandwidth).fit(X)

    for i in range(max_iter):
        print(f"Iteration: {i + 1}")
        
        new_centroids = np.zeros_like(centroids)
        for j in range(n_samples):
            # Find points within the bandwidth radius
            indices = nn.radius_neighbors([centroids[j]], return_distance=False)[0]
            
            # Compute the weighted mean of neighbors
            neighbors = centroids[indices]
            new_centroids[j] = np.mean(neighbors, axis=0)

        # Check for convergence
        shift = np.linalg.norm(new_centroids - centroids, axis=1)
        if np.all(shift < tol):
            print(f"Converged after {i + 1} iterations")
            break

        centroids = new_centroids

    return centroids


if __name__ == "__main__":
    image = io.imread('input.jpg')
    print("Image shape: ", image.shape)

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    save_image(image, title="Original Image")

    X = feature_space(image)

    # Run Mean Shift
    bandwidth = 0.1
    centroids = mean_shift(X, bandwidth=bandwidth)

    # Assign each pixel to the nearest centroid
    labels, _ = pairwise_distances_argmin_min(X, centroids)

    rows, cols, channels = image.shape

    # Reshape labels back to the original image shape
    segmented_image = labels.reshape(rows, cols)

    # Display the segmented image
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis('off')
    plt.show()
