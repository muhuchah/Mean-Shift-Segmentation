# Mean Shift Image Segmentation

This project implements the Mean Shift algorithm for image segmentation. The algorithm clusters pixels in an image based on both their color and spatial information, resulting in a segmented image where pixels with similar features are grouped together.
Features

- Feature Space Creation: Combines color and spatial information to create a 5D feature space.

- Mean Shift Clustering: Clusters pixels using the Mean Shift algorithm with customizable bandwidth and convergence criteria.

- Segmented Image Visualization: Displays the original and segmented images with real colors.

# Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `scikit-learn`
- `scikit-image`
- `matplotlib`

You can install the required libraries using pip:
```bash
pip install numpy scikit-learn scikit-image matplotlib
```

# Usage
Place your image in the same directory as the script and name it input.jpg.

Run the script:
```bash
python mean_shift.py
```

The script will:
- Display the original image.
- Perform Mean Shift clustering on the image.
- Display the segmented image with real colors.

# Code Structure

The project consists of the following functions:
- feature_space(image):
    - Converts the image into a 5D feature space (3 color channels + 2 normalized spatial coordinates).
    - Returns the feature matrix.

- mean_shift(X, bandwidth=0.1, max_iter=10, tol=1e-3):
    - Applies the Mean Shift algorithm to cluster pixels in the feature space.
    - Returns the final centroids.

- save_image(image, title="Image"):
    - Displays the image using matplotlib.

- Main Execution:
    - Reads the image, normalizes it, and creates the feature space.
    - Runs the Mean Shift algorithm and assigns pixels to the nearest centroid.
    - Displays the segmented image.

# Parameters
- bandwidth: Controls the size of the neighborhood for the Mean Shift algorithm. A smaller bandwidth results in more clusters.
- max_iter: Limits the number of iterations for the Mean Shift algorithm.
- tol: Determines the convergence criterion for the Mean Shift algorithm.

# Example Output
- Original Image:
    Original Image

- Segmented Image:
    Segmented Image

# Customization
To use a different image, replace input.jpg with your desired image file and update the filename in the script.
Adjust the bandwidth, max_iter, and tol parameters to fine-tune the segmentation results.

# License
This project is open-source and available under the MIT License.
Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

Enjoy experimenting with Mean Shift image segmentation! 🚀
