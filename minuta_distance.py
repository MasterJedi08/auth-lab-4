## Method 1 - Minuta distance
## This method attempts to extract the minuta from an image, then it multiplies the distance between extracted minuta together
## the product of the distances is the fingerprints ID and is stored for later comparison.

import cv2
import os
import numpy as np


def extractMinutiae(img):
    """
    Extract ridge endings and bifurcations from a fingerprint image.

    Parameters:
        img (numpy.ndarray): Grayscale fingerprint image.

    Returns:
        tuple: Two lists containing coordinates of ridge endings and bifurcations.
    """
    if img is None:
        raise ValueError("Invalid image provided.")

    # Push gray pixels to either black or white
    _, binaryImg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Perform skeletonization
    skeleton = cv2.ximgproc.thinning(binaryImg)

    # Find ridge ends and bifurcations with stricter sensitivity
    ridgeEnds = []
    bifurcations = []
    rows, cols = skeleton.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 255:  # Pixel is part of the skeleton
                # Extract the 3x3 neighborhood
                neighborhood = skeleton[i - 1:i + 2, j - 1:j + 2]
                white_pixels = np.sum(neighborhood == 255)

                # Apply stricter rules for ridge endings and bifurcations
                if white_pixels == 2:  # Ridge ending
                    ridgeEnds.append((j, i))
                elif white_pixels == 4:  # Bifurcation
                    bifurcations.append((j, i))

    # Remove duplicates and close-by minutiae
    ridgeEnds = filter_minutiae(ridgeEnds)
    bifurcations = filter_minutiae(bifurcations)

    return ridgeEnds, bifurcations


def filter_minutiae(minutiae, min_distance=10):
    """
    Filter minutiae to remove duplicates and close-by points.

    Parameters:
        minutiae (list): List of (x, y) coordinates of minutiae.
        min_distance (int): Minimum allowed distance between minutiae.

    Returns:
        list: Filtered minutiae.
    """
    filtered = []
    for point in minutiae:
        if all(np.linalg.norm(np.array(point) - np.array(p)) > min_distance for p in filtered):
            filtered.append(point)
    return filtered


def visualize_minutiae(img, ridge_ends, bifurcations):
    """
    Visualize ridge endings and bifurcations on the fingerprint image.

    Parameters:
        img (numpy.ndarray): Original fingerprint image.
        ridge_ends (list): List of (x, y) coordinates of ridge endings.
        bifurcations (list): List of (x, y) coordinates of bifurcations.
    """
    # Convert the grayscale image to a color image for visualization
    visual_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw ridge endings (in green)
    for x, y in ridge_ends:
        cv2.circle(visual_img, (x, y), 3, (0, 255, 0), -1)

    # Draw bifurcations (in red)
    for x, y in bifurcations:
        cv2.circle(visual_img, (x, y), 3, (0, 0, 255), -1)

    return visual_img


def main():
    # Define the folder containing fingerprint images
    folder_path = "train"

    # Verify the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    if not image_files:
        print(f"No image files found in folder '{folder_path}'.")
        return

    print(f"Found {len(image_files)} images in '{folder_path}'.")

    for image_file in image_files:
        # Load the fingerprint image
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Ensure image was loaded
        if img is None:
            print(f"Failed to load image: {image_file}")
            continue

        # Extract minutiae
        ridge_ends, bifurcations = extractMinutiae(img)

        # Visualize minutiae
        visual_img = visualize_minutiae(img, ridge_ends, bifurcations)

        # Display side-by-side images
        combined_img = np.hstack((cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), visual_img))
        cv2.imshow("Minutiae Detection - Original (Left) | Marked (Right)", combined_img)

        # Wait for user input to move to the next fingerprint
        print(f"Displaying minutiae for: {image_file}")
        print("Press any key to move to the next fingerprint...")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    print("All images processed.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


