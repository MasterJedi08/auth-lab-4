## Method 1 - Minuta distance
## This method attempts to extract the minuta from an image, then it multiplies the distance between extracted minuta together
## the product of the distances is the fingerprints ID and is stored for later comparison.

import cv2
import os
import numpy as np
from collections import defaultdict
from glob import glob


def enhance_fingerprint(img):
    """
    Enhance fingerprint image quality using various preprocessing techniques.

    Parameters:
        img (numpy.ndarray): Input grayscale fingerprint image

    Returns:
        numpy.ndarray: Enhanced fingerprint image
    """
    # Normalize image
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    enhanced = clahe.apply(normalized)

    # Reduce noise while preserving edges
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, searchWindowSize=21)

    # Enhance local contrast
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened


def get_crossing_number(neighborhood):
    """
    Calculate the crossing number for minutiae detection.

    Parameters:
        neighborhood (numpy.ndarray): 3x3 binary neighborhood

    Returns:
        int: Crossing number
    """
    pixels = neighborhood.flatten()
    # Remove center pixel
    pixels = np.delete(pixels, 4)
    # Calculate transitions
    transitions = np.sum(np.abs(pixels[:-1] - pixels[1:])) + abs(pixels[-1] - pixels[0])
    return transitions // 2


def claudeExtractMinutiae(img, min_distance=10):
    """
    Extract ridge endings and bifurcations from a fingerprint image.

    Parameters:
        img (numpy.ndarray): Grayscale fingerprint image (512x512)
        min_distance (int): Minimum distance between minutiae points

    Returns:
        tuple: Lists of ridge endings and bifurcations coordinates
    """
    if img is None or img.shape != (512, 512):
        raise ValueError("Invalid image: Must be 512x512 grayscale image")

    # Enhance image
    enhanced = enhance_fingerprint(img)
    cv2.imshow('enhahced', enhanced)

    # Binarize image
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,  # Larger block size for better adaptation
        -8
    )

    #cv2.imshow('binary', binary)

    # Perform skeletonization
    skeleton = cv2.ximgproc.thinning(binary)

    #cv2.imshow('skel', skeleton)

    # Initialize lists for minutiae
    ridge_endings = []
    bifurcations = []

    # Pad image to handle border pixels
    padded = np.pad(skeleton, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    return ridge_endings, bifurcations


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

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(9, 9))

    # Apply Gaussian blur to reduce noise
    img = cv2.fastNlMeansDenoising(img, None, h=10, searchWindowSize=21) #denoise
    img = cv2.GaussianBlur(img, (9, 9), 0)                               #blur
    img = clahe.apply(img)                                               #clahe

    # inversion is because bifurcations are breaks in the gaps,
    # I do it up here so I can process the images with different constituents
    blurred_inv = cv2.bitwise_not(img)

    # Use adaptive thresholding for binarization
    binaryImg = cv2.adaptiveThreshold(
        img,
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

    #cv2.imshow('bImg', binaryImg)
    #cv2.imshow('skele', skeleton)

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


def calculate_far_frr(scores, threshold):
    """Calculates FAR and FRR based on a threshold."""
    false_accepts = 0
    false_rejects = 0
    correct_guesses = 0
    wrong_guesses = 0

    for label, score_list in scores.items():
        for score, is_match in score_list:
            if is_match:  # Genuine comparison
                if score > threshold:
                    false_rejects += 1
                else:
                    correct_guesses += 1
            else:  # Impostor comparison
                if score <= threshold:
                    false_accepts += 1
                else:
                    wrong_guesses += 1

    total_genuine = sum([len([s for s, m in lst if m]) for lst in scores.values()])
    total_impostor = sum([len([s for s, m in lst if not m]) for lst in scores.values()])

    far = false_accepts / total_impostor if total_impostor else 0
    frr = false_rejects / total_genuine if total_genuine else 0

    return far, frr, correct_guesses, wrong_guesses


if __name__ == "__main__":
    train_dir  = "train"
    test_dir = "test"
    threshold = 0.065

    # Dictionary to store scores: {id: [(score, is_match)]}
    scores = defaultdict(list)

    # Load images
    reference_images = sorted(glob(os.path.join(train_dir, "f*.png")))
    subject_images = sorted(glob(os.path.join(train_dir, "s*.png")))

    # Match reference and subject images
    for ref_path in reference_images:
        ref_id = os.path.basename(ref_path).split('_')[0][1:]  # Extract ID
        ref_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref_minutiae, _ = extractMinutiae(ref_image)
        ref_descriptor = minutiae_to_descriptor(ref_minutiae)

        for subj_path in subject_images:
            subj_id = os.path.basename(subj_path).split('_')[0][1:]  # Extract ID
            subj_image = cv2.imread(subj_path, cv2.IMREAD_GRAYSCALE)
            subj_minutiae, _ = extractMinutiae(subj_image)
            subj_descriptor = minutiae_to_descriptor(subj_minutiae)

            # Calculate score
            score = np.linalg.norm(subj_descriptor - ref_descriptor)

            # Determine if it's a genuine or impostor comparison
            is_match = ref_id == subj_id
            scores[ref_id].append((score, is_match))

    # Calculate FAR, FRR, and guess counts
    far, frr, correct_guesses, wrong_guesses = calculate_far_frr(scores, threshold)

    # Output results
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"Correct Guesses: {correct_guesses}")
    print(f"Wrong Guesses: {wrong_guesses}")