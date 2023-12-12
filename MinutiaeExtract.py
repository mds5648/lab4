import os
import cv2
# import fingerprint_feature_extractor
import kid_named_finger as fingerprint_feature_extractor
from multiprocessing import Pool, Manager
import math
import numpy as np
import random

TRAINING_MAX = 5 #1500
TRAINING_MIN = 0 #0

TESTING_MAX = 7 #2000
TESTING_MIN = 6 #1501

PATH_TO_IMAGES = "./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt"

#Function to enable multiprocessing
def process_image(args):
    filename, images_dic, lock = args
    img = cv2.imread(filename, 0)
    if img is not None:
        print(filename.split("/")[-1])
        #The actual extraction function. For more info go to __init__.py in kid_named_finger
        FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img)
        with lock:
            images_dic[filename.split("/")[-1]] = (FeaturesBifurcations, FeaturesTerminations)

#takes in a range of files. Returns a dictionary 
#return has a dictionary of tuples (FeaturesTerm, FeaturesBif) which are lists of MinutiaeFeature objects
def read_in_image(min, max):
    with Manager() as manager:
        images_dic = manager.dict()
        lock = manager.Lock()
        with Pool() as pool:
            args = []
            for count in range(min, max + 1):
                folder = PATH_TO_IMAGES + "/figs_" + str(count)
                for filename in os.listdir(folder):
                    test = f"{folder}/{filename}"
                    args.append((test, images_dic, lock))
            pool.map(process_image, args) # multiprocessing stuff
        return dict(images_dic) 

#Takes in two fingers (the tuples) and a distance. More distance = more tolerance
def count_close(finger1, finger2, distance):
    count = 0
    for idx, curr_minutiae in enumerate(finger1[0]):
        for idx1, comp_minutiae in enumerate(finger2[0]):
            curr = curr_minutiae.locX, curr_minutiae.locY
            comp = comp_minutiae.locX, comp_minutiae.locY
            if math.dist(curr, comp) <= distance:
                count += 1
    for idx, curr_minutiae in enumerate(finger1[1]):
        for idx1, comp_minutiae in enumerate(finger2[1]):
            curr = curr_minutiae.locX, curr_minutiae.locY
            comp = comp_minutiae.locX, comp_minutiae.locY
            if math.dist(curr, comp) <= distance:
                count += 1
    return count

# Takes in the training dictionary of fingers and returns an average of close features
def train_compare(DA_DICT, dist, tolerance):
    close = []
    notClose = []
    s2 = None
    for key in DA_DICT.keys():
        if key[0] == "f":
            f = DA_DICT[key]
            s = DA_DICT[f's{key[1:]}']
            if s2 != None: 
                notClose.append(count_close(f,s2, dist))
            s2 = s
            # print(key[1:])
            close.append(count_close(f,s, dist))
    match, nonMatch = np.average(close), np.average(notClose)
    return math.floor(match-((match - nonMatch)*tolerance*3))

# Takes in the testing dictionary of fingers and a min_close value. 
# No return but prints out the FRR
def FRR(comp_dict, min_close, dist):
    pAss, fAil = 0, 0
    for key in comp_dict.keys():
        if key[0] == "f":
            f = comp_dict[key]
            s = comp_dict[f's{key[1:]}']
            if count_close(f,s, dist) >= min_close: # Counts how many close values are between the confirmed pairs
                pAss += 1
            else:
                fAil += 1
    print(f'FRR = {fAil/(pAss+fAil)*100}%')

# Takes in the testing dictionary of fingers and a min_close value
# No return but prints out the FAR
def FAR(comp_dict, min_close, dist):
    pAss, fAil, f_dict, s_dict = 0, 0, {}, {}
    for key in comp_dict.keys():
        if key[0] == "f":
            f_dict[key] = comp_dict[key]
        else:
            s_dict[key] = comp_dict[key]
    for key in f_dict.keys():
        f = comp_dict[key]
        rando_s = []
        #For time sake, only takes in 50 random incorrect fingers. 
        #Tries the correct finger against all of them
        while len(rando_s) < 50:
            choice = random.choice(list(s_dict.keys()))
            if choice[1:] != key[1:]:
                rando_s.append(choice)
        for choice in rando_s:
            s = s_dict[choice]
            if count_close(f,s, dist) >= min_close:
                fAil += 1
            else: 
                pAss += 1
    print(f'FAR = {fAil/(pAss+fAil)*100}%')
            

def test_compare(comp_dict, min_close, dist):
    FRR(comp_dict,min_close,dist)
    FAR(comp_dict,min_close,dist)
            
            
def test(DA_DICT, comp_dict, tolerance, dist = 10):
    min_close = train_compare(DA_DICT, dist, tolerance)
    print("Tolerance = " + str(tolerance))
    test_compare(comp_dict, min_close, dist)

if __name__ == '__main__':
    DA_DICT = read_in_image(TRAINING_MIN, TRAINING_MAX)
    comp_dict = read_in_image(TESTING_MIN, TESTING_MAX)
    for tol in range (0,11):
        test(DA_DICT, comp_dict, tol/10, dist=15)