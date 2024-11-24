"""
Hybrid Method - Minutiae Distance and SSIM

This method attempts to extract the minutiae features from an image, 
then it multiplies the distance between extracted minuta together
the product of the distances is stored for later comparison. The images
are also compared with SSIM. Both methods have to return true for
the fingerprint to be authenticated.
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray


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

def compute_ssim(img1, img2):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.

    Returns:
        float: The SSIM value between the two images.
    """
    return ssim(img1, img2, data_range=1.0)

# New function to combine both methods
def combined_fingerprint_matching(train_dir, subject_dir, ssim_threshold=0.8, distance_threshold=0.6):
    """
    Combines minutiae distance comparison and SSIM for fingerprint authentication.

    Args:
        train_dir (str): Path to the training directory.
        subject_dir (str): Path to the subject directory.
        ssim_threshold (float, optional): SSIM threshold for image similarity. Defaults to 0.8.
        distance_threshold (float, optional): Distance threshold for minutiae comparison. Defaults to 0.6.

    Returns:
        None
    """

    # Extract minutiae-based descriptors for training images
    train_descriptors = {}
    for train_file in os.listdir(train_dir):
        if train_file.endswith(".png"):
            img_path = os.path.join(train_dir, train_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            ridge_ends, _ = extractMinutiae(img)
            train_descriptors[train_file] = minutiae_to_descriptor(ridge_ends)

    # Process subject images and match using both methods
    for subject_file in os.listdir(subject_dir):
        if subject_file.endswith(".png"):
            subject_img_path = os.path.join(subject_dir, subject_file)
            subject_img = cv2.imread(subject_img_path, cv2.IMREAD_GRAYSCALE)
            subject_ridge_ends, _ = extractMinutiae(subject_img)
            subject_descriptor = minutiae_to_descriptor(subject_ridge_ends)

            subject_id = subject_file.split("_")[0]
            matched = False

            for train_file, train_descriptor in train_descriptors.items():
                train_id = train_file.split("_")[0]

                # Minutiae distance comparison
                distance_score = np.linalg.norm(subject_descriptor - train_descriptor)
                if distance_score < distance_threshold:
                    matched = True

                    # SSIM comparison
                    train_img_path = os.path.join(train_dir, train_file)
                    train_img = load_img(train_img_path)
                    ssim_score = compute_ssim(subject_img, train_img)
                    if ssim_score < ssim_threshold:
                        matched = False
                        break

                if matched:
                    print(f"Subject {subject_id} matched with {train_id}")
                    break

            if not matched:
                print(f"Subject {subject_id} not matched")

# Main function
if __name__ == "__main__":
    train_directory = "train"
    subject_directory = "subjects"
    combined_fingerprint_matching(train_directory, subject_directory)