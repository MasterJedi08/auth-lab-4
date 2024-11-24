"""
Hybrid Method - Minutiae Distance and SSIM

This method attempts to extract the minutiae features from an image, 
then it multiplies the distance between extracted minuta together
the product of the distances is stored for later comparison. The images
are also compared with SSIM. Both methods have to return true for
the fingerprint to be authenticated.
"""
import os
import numpy as np
import cv2

# Suppross Rounding Errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray
from scipy.spatial import distance



# SSIM-based Image Comparison (Method 1)
def load_image(image_path, target_size=(256, 256)):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        return rgb2gray(img_array)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def compute_ssim(image1, image2):
    similarity = ssim(image1, image2, data_range=1.0)
    return similarity

def load_image_pairs(image_dir):
    all_files = os.listdir(image_dir)
    png_files = [file for file in all_files if file.endswith('.png')]

    f_images = [file.split('.')[0][1:] for file in png_files if file.startswith('f')]
    s_images = [file.split('.')[0][1:] for file in png_files if file.startswith('s')]

    pair_list = list(set(f_images).intersection(set(s_images)))

    images = []
    for pair in pair_list:
        fi_image = load_image(image_dir + "/" + f"f{pair}.png")
        si_image = load_image(image_dir + "/" + f"s{pair}.png")
        if fi_image is not None and si_image is not None:
            images.append((fi_image, si_image))
        else:
            print(f"skip pair {pair} - failed image load")

    train_files = images[700:750] # :750
    test_files = images[750:800] # 750:1500
    return train_files,test_files

def evaluate_template_matching(images, threshold=0.15):
    correct_predictions = 0
    total_predictions = len(images)
    invalid_matches = 0

    for fi_image, si_image in images:
        ssim_score = compute_ssim(fi_image, si_image)
        if ssim_score >= threshold:
            correct_predictions += 1
        else:
            invalid_matches += 1

    accuracy = correct_predictions / total_predictions
    frr = invalid_matches / total_predictions
    return accuracy, frr

# Function to evaluate minutiae matching (Accuracy and False Rejection Rate)
def evaluate_minutiae_matching(train_images, test_images, minutiae_threshold=0.6):
    correct_predictions = 0
    total_predictions = len(train_images)
    invalid_matches = 0

    for train_fi, train_si in train_images:
        print(f"Processing image pair: Correct Predictions {correct_predictions}")
        # Extract minutiae for the training image (template) and test image
        ridge_ends_fi, _ = extractMinutiae(train_fi)
        ridge_ends_si, _ = extractMinutiae(train_si)

        # Convert minutiae to descriptors
        fi_descriptor = minutiae_to_descriptor(ridge_ends_fi)
        si_descriptor = minutiae_to_descriptor(ridge_ends_si)

        # Compare the minutiae descriptors using Euclidean distance
        match = match_fingerprints(fi_descriptor, si_descriptor, minutiae_threshold)

        if match:
            correct_predictions += 1
        else:
            invalid_matches += 1

    # Calculate the accuracy and false rejection rate (FRR)
    accuracy = correct_predictions / total_predictions
    frr = invalid_matches / total_predictions

    return accuracy, frr

# Minutiae Distance Comparison (Method 2)
def extractMinutiae(img):
    try:
        if img is None:
            raise ValueError("Invalid image provided.")
            
        # Ensure the image is grayscale and of type uint8
        if len(img.shape) != 2:  # Not grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        blurred_inv = cv2.GaussianBlur(img, (9, 9), 0)
        blurred_inv = cv2.bitwise_not(blurred_inv)

        binaryImg = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, -8)
        binaryImg_inv = cv2.adaptiveThreshold(
            blurred_inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, -5)

        skeleton = cv2.ximgproc.thinning(binaryImg)
        skeleton_inv = cv2.ximgproc.thinning(binaryImg_inv)

        ridgeEnds = []
        bifurcations = []
        rows, cols = skeleton.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if skeleton[i, j] == 255:
                    neighborhood = skeleton[i - 2:i + 3, j - 2:j + 3]
                    white_pixels = np.sum(neighborhood == 255)
                    if white_pixels <= 3:
                        continue
                    elif white_pixels <= 4:
                        ridgeEnds.append((j, i))

        rows, cols = skeleton_inv.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if skeleton_inv[i, j] == 255:
                    neighborhood = skeleton_inv[i - 2:i + 3, j - 2:j + 3]
                    white_pixels = np.sum(neighborhood == 255)
                    if white_pixels <= 3:
                        continue
                    elif white_pixels <= 4:
                        bifurcations.append((j, i))

        ridgeEnds = filter_minutiae(ridgeEnds)
        bifurcations = filter_minutiae(bifurcations)

        return ridgeEnds, bifurcations
    
    except Exception as e:
        print(f"Error extracting minutiae from image: {e}")
        return [],[]

def filter_minutiae(minutiae, min_distance=26):
    filtered = []
    for point in minutiae:
        if all(np.linalg.norm(np.array(point) - np.array(p)) > min_distance for p in filtered):
            filtered.append(point)
    return filtered

def minutiae_to_descriptor(minutiae, scale_factor=100, histogram_bins=8):
    if len(minutiae) < 2:
        minutiae += [0, 0], [1, 0], [0, 1]

    minutiae = np.array(minutiae)
    mean_x, mean_y = np.mean(minutiae, axis=0)
    normalized = minutiae - [mean_x, mean_y]

    descriptors = []
    for i, (x1, y1) in enumerate(normalized):
        for j, (x2, y2) in enumerate(normalized):
            if i >= j:
                continue
            dist = np.linalg.norm([x2 - x1, y2 - y1]) / scale_factor
            angle = np.arctan2(y2 - y1, x2 - x1)
            angle = angle if angle >= 0 else angle + 2 * np.pi
            descriptors.append((dist, angle))

    descriptors = np.array(descriptors)
    distance_hist, _ = np.histogram(descriptors[:, 0], bins=histogram_bins, range=(0, 1))
    angle_hist, _ = np.histogram(descriptors[:, 1], bins=histogram_bins, range=(0, 2 * np.pi))

    fingerprint_descriptor = np.concatenate([distance_hist, angle_hist])
    fingerprint_descriptor = fingerprint_descriptor / np.sum(fingerprint_descriptor)
    return fingerprint_descriptor

def match_fingerprints(features_1, features_2, threshold=0.6):
    distance = np.linalg.norm(features_1 - features_2)
    return distance < threshold

# Combined Evaluation
def combined_evaluation(train_images, test_images, threshold_minutiae=0.6, threshold_ssim=0.15):
    minutiae_accuracy = 0
    ssim_accuracy = 0
    minutiae_frr = 0
    ssim_frr = 0

    try:
        print("Evaluating Minutiae Matching...")
        minutiae_accuracy, minutiae_frr = evaluate_minutiae_matching(train_images, threshold_minutiae)
        print("Evaluating SSIM Matching...")
        ssim_accuracy, ssim_frr = evaluate_template_matching(train_images, threshold_ssim)

        print(f"Minutiae Matching - Accuracy: {minutiae_accuracy:.4f}, FRR: {minutiae_frr:.4f}")
        print(f"SSIM Matching - Accuracy: {ssim_accuracy:.4f}, FRR: {ssim_frr:.4f}")

        return minutiae_accuracy, minutiae_frr, ssim_accuracy, ssim_frr
    except Exception as e:
        print(f"Error during combined evaluation: {e}")
        return None, None, None, None

# Main function to load data and run evaluation
def main():
    img_dir = 'C:/Users/koral/Documents/PersonalPapers/SchoolworkFall24/Auth/test_imgs'  # path to your image directory

    print("Loading image pairs...")
    train_images, test_images = load_image_pairs(img_dir)

    # Combined evaluation using both SSIM and Minutiae matching
    combined_evaluation(train_images, test_images)

if __name__ == "__main__":
    main()
