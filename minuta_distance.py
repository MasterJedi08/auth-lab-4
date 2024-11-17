## Method 1 - Minuta distance
## This method attempts to extract the minuta from an image, then it multiplies the distance between extracted minuta together
## the product of the distances is the fingerprints ID and is stored for later comparison.

import cv2
import os
import numpy as np
from scipy.spatial import distance


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

    cv2.imshow('binary', binary)

    # Perform skeletonization
    skeleton = cv2.ximgproc.thinning(binary)

    cv2.imshow('skel', skeleton)

    # Initialize lists for minutiae
    ridge_endings = []
    bifurcations = []

    # Pad image to handle border pixels
    padded = np.pad(skeleton, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Scan the image
    rows, cols = skeleton.shape
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if padded[i, j] == 255:  # Ridge pixel
                # Get 3x3 neighborhood
                neighborhood = padded[i - 1:i + 2, j - 1:j + 2]
                cn = get_crossing_number(neighborhood)

                # cn = 1 for ridge ending
                # cn = 3 for bifurcation
                if cn == 1:
                    ridge_endings.append((j - 1, i - 1))
                elif cn == 3:
                    bifurcations.append((j - 1, i - 1))

    def filter_close_points(points, min_dist):
        """Remove points that are too close to each other"""
        if not points:
            return points

        filtered = [points[0]]
        for point in points[1:]:
            if all(np.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) >= min_dist
                   for p in filtered):
                filtered.append(point)
        return filtered

    # Filter out minutiae that are too close to each other
    ridge_endings = filter_close_points(ridge_endings, min_distance)
    bifurcations = filter_close_points(bifurcations, min_distance)

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


def fingerprint_matching(train_dir, subject_dir, threshold=0.01):
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
        cv2.circle(visual_img, (x, y), 1, (0, 255, 0), -1)

    # Draw bifurcations (in red)
    for x, y in bifurcations:
        cv2.circle(visual_img, (x, y), 1, (0, 0, 255), -1)

    return visual_img

def testMatching(train_dir, subject_dir, threshold=0.01):
    fRidge_ends = []
    fBifurcations = []
    sRidge_ends = []
    sBifurcations = []


    # Extract descriptors for training images
    print("Processing training images...")
    for train_file in os.listdir(train_dir):
        if train_file.endswith(".png") and train_file[0] == 'f':  # Assuming PNG format
            fImg_path = os.path.join(train_dir, train_file)
            fimg = cv2.imread(fImg_path, cv2.IMREAD_GRAYSCALE)
            fRidge_ends, fBifurcations = claudeExtractMinutiae(fimg)

            sTrain_file = 's' + train_file[1:]
            sImg_path = os.path.join(train_dir, sTrain_file)
            simg = cv2.imread(sImg_path, cv2.IMREAD_GRAYSCALE)
            sRidge_ends, sBifurcations = claudeExtractMinutiae(simg)

            # Display side-by-side images
            fVis = visualize_minutiae(fimg, fRidge_ends, fBifurcations)
            fCombined_img = np.hstack((cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR), fVis))
            cv2.imshow("%s - Original (L) | Vis (R)" % train_file, fCombined_img)

            sVis = visualize_minutiae(simg, sRidge_ends, sBifurcations)
            sCombined_img = np.hstack((cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR), sVis))
            cv2.imshow("%s - Original (L) | Vis (R)" % sTrain_file, sCombined_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    train_directory = "train"
    subject_directory = "subjects"
    testMatching(train_directory, subject_directory, threshold=0.001)