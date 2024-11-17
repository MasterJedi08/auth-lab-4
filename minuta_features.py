import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
from skimage.morphology import skeletonize
import os
import random

# minutiae features extraction
def extract_minutiae_points(image_path):
    # extract ridge endings and bifurcations from the skeletonized fingerprint image

    # make sure file exists
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Unable to load image: {image_path}")
        return None

    # preprosccing the image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary > 0)

    minutiae_points = []
    rows, cols = skeleton.shape

    # extract the minutiae points
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j]:
                neighborhood = skeleton[i-1:i+2, j-1:j+2]
                count = np.sum(neighborhood)

                if count == 2:  # Ridge ending
                    minutiae_points.append((i, j))
                elif count == 4:  # Bifurcation
                    minutiae_points.append((i, j))

    print(f"Extracted {len(minutiae_points)} minutiae from {image_path}")
    return np.array(minutiae_points)

#minutiae similarity calculation
def calculate_minutiae_similarity(minutiae_f, minutiae_s):
  
    # calcs the similarity between two sets of minutiae points using nearest neighbor distances
    
    if minutiae_f is None or minutiae_s is None or minutiae_f.size == 0 or minutiae_s.size == 0:
        print("Empty minutiae points, returning infinite similarity.")
        return float('inf')

    distances = pairwise_distances(minutiae_f, minutiae_s, metric='euclidean')
    similarity = np.mean(np.min(distances, axis=1))  # Mean of nearest distances
    print(f"Calculated similarity: {similarity}")
    return similarity

# simulate impostor pairs
def simulate_impostor_pairs(train_pairs):
  
    #simulate impostor pairs by shuffling subject images
  
    shuffled_pairs = random.sample(train_pairs, len(train_pairs))
    impostor_pairs = [(f_path, s_path) for (f_path, _), (_, s_path) in zip(train_pairs, shuffled_pairs)]
    print(f"Generated {len(impostor_pairs)} impostor pairs.")
    return impostor_pairs

# stats calculation
def calculate_statistics(train_pairs, impostor_pairs, threshold=10.0):

    # calcs average, min, and max FRR and FAR, and EER for the given pairs
    
    fr_rates = []
    fa_rates = []
    false_rejects = false_accepts = total_matches = total_non_matches = 0

    for f_path, s_path in train_pairs + impostor_pairs:
        minutiae_f = extract_minutiae_points(f_path)
        minutiae_s = extract_minutiae_points(s_path)

        if minutiae_f is None or minutiae_s is None:
            continue

        similarity = calculate_minutiae_similarity(minutiae_f, minutiae_s)
        is_match = (f_path, s_path) in train_pairs

        if similarity <= threshold:
            if is_match:
                total_matches += 1  # True positive
            else:
                false_accepts += 1  # False accept
        else:
            if is_match:
                false_rejects += 1  # False reject
            else:
                total_non_matches += 1  # True negative

        # Individual FRR and FAR
        if total_matches + false_rejects > 0:
            fr_rate = false_rejects / (total_matches + false_rejects)
        else:
            fr_rate = 0

        if total_non_matches + false_accepts > 0:
            fa_rate = false_accepts / (total_non_matches + false_accepts)
        else:
            fa_rate = 0

        fr_rates.append(fr_rate)
        fa_rates.append(fa_rate)

    # calc statistics
    if fr_rates and fa_rates:
        frr_avg, frr_min, frr_max = np.mean(fr_rates), np.min(fr_rates), np.max(fr_rates)
        far_avg, far_min, far_max = np.mean(fa_rates), np.min(fa_rates), np.max(fa_rates)
        eer = (frr_avg + far_avg) / 2

        print("Method: Minutiae Detection")
        print(f"FRR Avg: {frr_avg:.4f}, FRR Min: {frr_min:.4f}, FRR Max: {frr_max:.4f}")
        print(f"FAR Avg: {far_avg:.4f}, FAR Min: {far_min:.4f}, FAR Max: {far_max:.4f}")
        print(f"EER: {eer:.4f}")

        return {
            "FRR Avg": frr_avg,
            "FRR Min": frr_min,
            "FRR Max": frr_max,
            "FAR Avg": far_avg,
            "FAR Min": far_min,
            "FAR Max": far_max,
            "EER": eer,
        }
    else:
        print("No valid statistics calculated - check inputs")
        return None


def main():
    train_pairs = []
    for i in range(1, 1500):  # loop over TRAIN 
        for j in range(1, 11):  # loop over suffixes 1 through 10
            f_path = f"TRAIN/f{i:04d}_{j:02d}.png"
            s_path = f"TRAIN/s{i:04d}_{j:02d}.png"
            if os.path.exists(f_path) and os.path.exists(s_path):
                train_pairs.append((f_path, s_path))
            else:
                print(f"Missing file for pair: {f_path}, {s_path}")

    if not train_pairs:
        print("No valid TRAIN pairs found.")
        return

    # simulates impostor pairs
    impostor_pairs = simulate_impostor_pairs(train_pairs)

    # calcs and display statistics
    stats = calculate_statistics(train_pairs, impostor_pairs, threshold=10.0)

    if stats:
        print("\nSummary of Results:")
        print(f"FRR Avg: {stats['FRR Avg']:.4f}, FRR Min: {stats['FRR Min']:.4f}, FRR Max: {stats['FRR Max']:.4f}")
        print(f"FAR Avg: {stats['FAR Avg']:.4f}, FAR Min: {stats['FAR Min']:.4f}, FAR Max: {stats['FAR Max']:.4f}")
        print(f"EER: {stats['EER']:.4f}")


if __name__ == "__main__":
    main()



"""


Cited 
(chatgpt was used as a resource for writing this code)

OpenAI. ChatGPT (November 2024 Version). OpenAI, 2024, https://chat.openai.com/.

Prompts used included: 
- Explain how to extract minutiae points and use featured based implementation
- Explain preprocessing fingerprint images and how to implement it 
- Explain feature matching and imposter usage and implementation
- best way to implement FAR and FRR (avg, minm max) and EER statistics
- Explain FAR and FRR statitstics and how to improve data collection and statiscitcs
- Describe thresholds and best methods


"""
