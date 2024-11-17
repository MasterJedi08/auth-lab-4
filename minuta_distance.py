## Method 1 - Minuta distance
## This method attempts to extract the minuta from an image, then it multiplies the distance between extracted minuta together
## the product of the distances is the fingerprints ID and is stored for later comparison.

import cv2
import os
import numpy as np
from scipy.spatial import distance


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

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # inversion is because bifurcations are breaks in the gaps,
    # I do it up here so I can process the images with different constituents
    blurred_inv = cv2.GaussianBlur(img, (9, 9), 0)

    blurred_inv = cv2.bitwise_not(blurred_inv)

    # Use adaptive thresholding for binarization
    binaryImg = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,  # Block size (must be odd, e.g., 11 or 15)
        -8  # Constant subtracted from mean
    )

    binaryImg_inv = cv2.adaptiveThreshold(
        blurred_inv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,  # Block size (must be odd, e.g., 11 or 15)
        -5  # Constant subtracted from mean
    )

    # Perform skeletonization
    skeleton = cv2.ximgproc.thinning(binaryImg)
    skeleton_inv = cv2.ximgproc.thinning(binaryImg_inv)

    # Find ridge ends and bifurcations with stricter sensitivity
    ridgeEnds = []
    bifurcations = []
    rows, cols = skeleton.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 255:  # Pixel is part of the skeleton
                # Extract the 5 radius neighborhood
                neighborhood = skeleton[i - 2:i + 3, j - 2:j + 3]
                white_pixels = np.sum(neighborhood == 255)

                if white_pixels <= 3:
                    # false positive, white pixel from noise
                    continue
                elif white_pixels <= 4:  # Ridge ending
                    ridgeEnds.append((j, i))

    rows, cols = skeleton_inv.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton_inv[i, j] == 255:  # Pixel is part of the skeleton
                # Extract the 5 radius neighborhood
                neighborhood = skeleton_inv[i - 2:i + 3, j - 2:j + 3]
                white_pixels = np.sum(neighborhood == 255)

                if white_pixels <= 3:
                    continue
                elif white_pixels <= 4:  # Ridge ending
                    bifurcations.append((j, i))

    # Remove duplicates and close-by minutiae
    ridgeEnds = filter_minutiae(ridgeEnds)
    bifurcations = filter_minutiae(bifurcations)

    return ridgeEnds, bifurcations


def filter_minutiae(minutiae, min_distance=26):
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


# Function to match fingerprints based on their feature vectors
def match_fingerprints(features_1, features_2, threshold=0.6):
    distance = np.linalg.norm(features_1 - features_2)
    return distance < threshold


def minutiae_to_descriptor(minutiae, scale_factor=100, histogram_bins=8):
    """
    Convert a set of minutiae points into a small array of numbers for matching.

    Parameters:
        minutiae (list of tuples): List of (x, y) coordinates of minutiae points.
        scale_factor (int): Factor to normalize distances for histogram computation.
        histogram_bins (int): Number of bins for the angular and distance histograms.

    Returns:
        numpy.ndarray: Descriptor array representing the fingerprint minutiae.
    """
    if len(minutiae) < 2:
        minutiae += [0,0], [1,0], [0,1] ##todo, error handling

    # Normalize the minutiae to have mean 0 (translation invariance)
    minutiae = np.array(minutiae)
    mean_x, mean_y = np.mean(minutiae, axis=0)
    normalized = minutiae - [mean_x, mean_y]

    # Compute pairwise distances and angles between minutiae
    descriptors = []
    for i, (x1, y1) in enumerate(normalized):
        for j, (x2, y2) in enumerate(normalized):
            if i >= j:  # Avoid duplicate pairs
                continue
            # Distance between points
            dist = np.linalg.norm([x2 - x1, y2 - y1]) / scale_factor
            # Angle between points (in radians)
            angle = np.arctan2(y2 - y1, x2 - x1)
            # Wrap angle to [0, 2*pi]
            angle = angle if angle >= 0 else angle + 2 * np.pi
            descriptors.append((dist, angle))

    # Convert descriptors into histograms
    descriptors = np.array(descriptors)
    if descriptors.size == 0:
        return np.zeros(histogram_bins * 2)

    # Distance histogram
    distance_hist, _ = np.histogram(descriptors[:, 0], bins=histogram_bins, range=(0, 1))
    # Angle histogram
    angle_hist, _ = np.histogram(descriptors[:, 1], bins=histogram_bins, range=(0, 2 * np.pi))

    # Concatenate histograms into a single descriptor
    fingerprint_descriptor = np.concatenate([distance_hist, angle_hist])

    # Normalize the descriptor to sum to 1 (optional for matching robustness)
    fingerprint_descriptor = fingerprint_descriptor / np.sum(fingerprint_descriptor)

    return fingerprint_descriptor


def fingerprint_matching(train_dir, subject_dir, threshold=0.7):
    train_descriptors = {}
    total_comparisons = 0
    total_accepts = 0
    total_rejects = 0
    false_accepts = 0
    false_rejects = 0
    true_accepts = 0
    true_rejects = 0

    # Extract descriptors for training images
    print("Processing training images...")
    for train_file in os.listdir(train_dir):
        if train_file.endswith(".png"):  # Assuming PNG format
            img_path = os.path.join(train_dir, train_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            ridge_ends, _ = extractMinutiae(img)
            train_descriptors[train_file] = minutiae_to_descriptor(ridge_ends)
            #print(train_file, minutiae_to_descriptor(ridge_ends))

    print("Processing subject images and matching...")
    for subject_file in os.listdir(subject_dir):
        if subject_file.endswith(".png"):
            img_path = os.path.join(subject_dir, subject_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            ridge_ends, _ = extractMinutiae(img)
            subject_descriptor = minutiae_to_descriptor(ridge_ends)

            subject_id = subject_file.split("_")[0]
            matched = False
            for train_file, train_descriptor in train_descriptors.items():
                train_id = train_file.split("_")[0]
                score = np.linalg.norm(subject_descriptor - train_descriptor)
                total_comparisons += 1
                if score < threshold:
                    total_accepts += 1
                    matched = True
                    if train_id == subject_id:
                        true_accepts += 1
                    else:
                        false_accepts += 1
                else:
                    total_rejects += 1
                    if train_id == subject_id:
                        false_rejects += 1
                    else:
                        true_rejects += 1

            if not matched and subject_id in train_descriptors:
                false_rejects += 1

            # Periodic reporting
            if total_comparisons % 10000 == 0:
                tar = true_accepts / (true_accepts + false_rejects) if true_accepts + false_rejects > 0 else 0
                trr = true_rejects / (true_rejects + false_accepts) if true_rejects + false_accepts > 0 else 0
                print(f"Comparisons: {total_comparisons}")
                print(f"False Accept Rate: {false_accepts / total_comparisons:.4f}")
                print(f"False Reject Rate: {false_rejects / total_comparisons:.4f}")
                print(f"True Accept Rate: {tar:.4f}")
                print(f"True Reject Rate: {trr:.4f}")
                print(f"Total Accepts: {total_accepts}")
                print(f"Total Rejects: {total_rejects}")

    # Final statistics
    tar = true_accepts / (true_accepts + false_rejects) if true_accepts + false_rejects > 0 else 0
    trr = true_rejects / (true_rejects + false_accepts) if true_rejects + false_accepts > 0 else 0

    print("\nFinal Statistics:")
    print(f"Comparisons: {total_comparisons}")
    print(f"False Accept Rate: {false_accepts / total_comparisons:.4f}")
    print(f"False Reject Rate: {false_rejects / total_comparisons:.4f}")
    print(f"True Accept Rate: {tar:.4f}")
    print(f"True Reject Rate: {trr:.4f}")
    print(f"Total Accepts: {total_accepts}")
    print(f"Total Rejects: {total_rejects}")


if __name__ == "__main__":
    train_directory = "train"
    subject_directory = "subjects"
    fingerprint_matching(train_directory, subject_directory, threshold=0.001)