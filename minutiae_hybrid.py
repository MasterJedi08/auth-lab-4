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



def load_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Failed to load image")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def process_image(img):
    try:
        if len(img.shape) == 2:
            return img
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ... further processing
            return gray_img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def load_image_pairs(image_dir):
    """Loads image pairs from a directory.

    Args:
        image_dir: The directory containing the images.

    Returns:
        A list of image pairs.
    """

    all_files = os.listdir(image_dir)
    png_files = [file for file in all_files if file.endswith('.png')]

    # Get the file names without extensions and remove the first character
    f_images = [file.split('.')[0][1:] for file in png_files if file.startswith('f')]
    s_images = [file.split('.')[0][1:] for file in png_files if file.startswith('s')]

    pair_list = list(set(f_images).intersection(set(s_images)))

    images = []
    for pair in pair_list:
        # Use os.path.join to create paths in a cross-platform way
        fi_path = os.path.join(image_dir, f"f{pair}.png")
        si_path = os.path.join(image_dir, f"s{pair}.png")

        try:
            fi_img = cv2.imread(fi_path, cv2.IMREAD_GRAYSCALE)
            si_img = cv2.imread(si_path, cv2.IMREAD_GRAYSCALE)

            if fi_img is None or si_img is None:
                raise ValueError(f"Failed to load image pair: {fi_path}, {si_path}")

            # Process images (e.g., normalization, filtering)
            fi_img = process_image(fi_img)
            si_img = process_image(si_img)

            images.append((fi_img, si_img))

        except Exception as e:
            print(f"Error processing image pair {pair} ({fi_path}, {si_path}): {e}")

    return images

# SSIM-based Image Comparison (Method 1)
def compute_ssim(img1, img2):
    """Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.

    Returns:
        The SSIM value.
    """
    # Preprocessing
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)  # Apply Gaussian blur
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    # Normalize images to 0-1 range
    img1 = img1 / 255.0
    img2 = img2 / 255.0

    # Ensure images are grayscale
    if len(img1.shape) == 3:
        img1 = rgb2gray(img1)
    if len(img2.shape) == 3:
        img2 = rgb2gray(img2)

    return ssim(img1, img2, data_range=1.0)

# Minutiae 
def extract_minutiae(img):
    """Extracts minutiae points from a fingerprint image using OpenCV.

    Args:
        img: The input fingerprint image.

    Returns:
        A list of minutiae points (x, y) coordinates.
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Create a FeatureDetector object
    fd = cv2.SIFT_create()

    # Detect keypoints
    kp = fd.detect(img, None)

    # Extract minutiae points
    minutiae = []
    for point in kp:
        x, y = int(point.pt[0]), int(point.pt[1])
        minutiae.append((x, y))

    return minutiae

def match_minutiae(minutiae1, minutiae2, threshold=0.7):
    """Matches two sets of minutiae points.

    Args:
        minutiae1: The first set of minutiae points.
        minutiae2: The second set of minutiae points.
        threshold: The distance threshold for matching.

    Returns:
        True if the two sets of minutiae match, False otherwise.
    """

    # Implement a suitable matching algorithm, such as:
    # - Brute-force matching
    # - Feature-based matching using descriptors
    # - Graph-based matching

    # Here's a simple example using brute-force matching:
    for p1 in minutiae1:
        for p2 in minutiae2:
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            if distance < threshold:
                return True
    return False

def evaluate_hybrid_method(images, ssim_threshold=0.2, minutiae_threshold=0.17):
    """Evaluates the hybrid method on a given dataset.

    Args:
        train_images: A list of training image pairs.
        test_images: A list of test image pairs.
        ssim_threshold: The SSIM threshold for matching.
        minutiae_threshold: The minutiae matching threshold.

    Returns:
        A tuple containing the true accept rate, true reject rate, false accept rate, false reject rate, total accepts, and total rejects.
    """
    total_accept = 0
    total_reject = 0
    true_accepts = 0
    true_rejects = 0
    false_accepts = 0
    false_rejects = 0

    try:
        for (img1, img2) in images:
            ssim_score = compute_ssim(img1, img2)
            minutiae_match = match_minutiae(extract_minutiae(img1), extract_minutiae(img2), minutiae_threshold)

            if ssim_score >= ssim_threshold and minutiae_match:
                if ssim_score < 0.23:
                    false_accepts +=1
                else:
                    true_accepts += 1
                total_accept += 1
            else:
                # False Reject
                if ssim_score > minutiae_threshold:
                    false_rejects += 1
                else:
                    true_rejects += 1
                total_reject +=1

        tar = true_accepts / (true_accepts + false_rejects) if true_accepts + false_rejects > 0 else 0
        trr = true_rejects / (true_rejects + false_accepts) if true_rejects + false_accepts > 0 else 0

        print("\nFinal Statistics:")
        print(f"Comparisons: {total_accept + total_reject}")
        print(f"False Accept Rate: {false_accepts / total_accept:.4f}")
        print(f"False Reject Rate: {false_rejects / total_reject:.4f}")
        print(f"True Accept Rate: {tar:.4f}")
        print(f"True Reject Rate: {trr:.4f}")
        print(f"Total Accepts: {total_accept}")
        print(f"Total Rejects: {total_reject}")

    except Exception as e:
        print(f"Error during evaluation: {e}")

# Main function to load data and run evaluation
def main():
    img_dir = 'C:/Users/koral/Documents/PersonalPapers/SchoolworkFall24/Auth/test_imgs'  # path to your image directory

    print("Loading image pairs...")
    images = load_image_pairs(img_dir)

    # Combined evaluation using both SSIM and Minutiae matching
    print("Evaluating hybrid method...")
    evaluate_hybrid_method(images)


if __name__ == "__main__":
    main()
