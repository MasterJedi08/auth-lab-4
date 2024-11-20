# method 2 - basic image comparison with SSIM

# imports
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray

# helper functions for loading/preprocessing images
def load_image(image_path, target_size=(256, 256)):
    try:
        img = load_img(image_path, target_size=target_size)
        # normalize pixel values 
        img_array = img_to_array(img) / 255.0  
        # convert to grayscale ik its already in grayscale but it breaks without this
        return rgb2gray(img_array)  
        #return img_array

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None 

# load img pairs (s/f) and split into training and testing data
def load_image_pairs(image_dir):
    # list of all files in the img directory
    all_files = os.listdir(image_dir)

    # filter list to include only files that end with .png
    png_files = [file for file in all_files if file.endswith('.png')]

    #print(png_files)
    # split filenames into two groups since theyre matching pairs: f and s
    f_images = [file.split('.')[0][1:] for file in png_files if file.startswith('f')]
    s_images = [file.split('.')[0][1:] for file in png_files if file.startswith('s')]

    # pair_list is just the #s
    pair_list = list(set(f_images).intersection(set(s_images)))

    images = []
    for pair in pair_list:
        # print("pair:", pair)
        fi_image = load_image(image_dir + "/" + f"f{pair}.png")
        si_image = load_image(image_dir + "/" + f"s{pair}.png")

        # make sure both images are loaded properly
        if fi_image is not None and si_image is not None:
            images.append((fi_image, si_image))
        else:
            print(f"skip pair {pair} - failed image load")
    
    print(len(images))
    train_files = images[:750]
    test_files = images[750:1500]
    return train_files,test_files

# ----

# SSIM similarity for 2 imgs
def compute_ssim(image1, image2):
    #print("computing ssim")
    similarity = ssim(image1, image2, data_range=1.0)
    #print("similarity: ", similarity)
    return(similarity)

# goes through and teplate matches all img pairs
def evaluate_template_matching(train_images, threshold=0.15):
    correct_predictions = 0
    total_predictions = len(train_images)
    #print("in evaluate")
    invalid_matches = 0
    
    for train_fi, train_si in train_images:
        #print("in for loop")
        # compute ssim for both fi and si images
        ssim = compute_ssim(train_fi, train_si)

        # classify based on ssim score
        if ssim >= threshold:
            correct_predictions += 1
        else:
            invalid_matches += 1
    
    #print(f"values: {correct_predictions} / {total_predictions}")

    
    
    accuracy = correct_predictions / total_predictions
    frr = invalid_matches/total_predictions    

    return accuracy, frr

# ------

def main():    
    # Path to the parent directory containing subfolders
    img_dir = 'C:/Users/x/auth-lab-4/test_imgs'    

    # Load image pairs
    print("loading image pairs ...")
    train_images, test_images = load_image_pairs(img_dir)

    #print(len(train_images), " - ", len(test_images))
    # print("train imgs: ", train_images)

    frr = []
    accuracy = []

    print("SSIM matching image pairs ... ")
    # template matching (SSIM) accuracy
    ssim_accuracy, round_frr = evaluate_template_matching(train_images)

    frr.append(round_frr)
    accuracy.append(ssim_accuracy)

    # Results
    print(f"----\nRound 1 \nFRR: {round_frr}")
    print(f"Template Matching (SSIM) Accuracy: {ssim_accuracy:.4f}\n")

    print("SSIM matching TEST image pairs ... ")
    # template matching (SSIM) accuracy
    ssim_accuracy, round_frr = evaluate_template_matching(test_images)

    frr.append(round_frr)
    accuracy.append(ssim_accuracy)

    # Results
    print(f"----\nRound 1 \nFRR: {round_frr}")
    print(f"Template Matching (SSIM) Accuracy: {ssim_accuracy:.4f}\n")

    print("\nRESULTS - FRR AVG: ", (sum(frr)/len(frr)), " / AVG Accuracy: ", (sum(accuracy)/len(accuracy)))

main() 