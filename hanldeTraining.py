import os
import cv2
# import fingerprint_feature_extractor
import kid_named_finger as fingerprint_feature_extractor
from multiprocessing import Pool, Manager
import math
import numpy as np

TRAINING_MAX = 5 #1500
TRAINING_MIN = 0 #0

TESTING_MAX = 7 #2000
TESTING_MIN = 6 #1501

PATH_TO_IMAGES = "./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt"

def process_image(args):
    filename, images_dic, lock = args
    img = cv2.imread(filename, 0)
    if img is not None:
        print(filename.split("/")[-1])
        FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img)
        with lock:
            images_dic[filename.split("/")[-1]] = (FeaturesBifurcations, FeaturesTerminations)

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
            pool.map(process_image, args)
        return dict(images_dic)
        
def count_close(finger1, finger2):
    count = 0
    for idx, curr_minutiae in enumerate(finger1[0]):
        for idx1, comp_minutiae in enumerate(finger2[0]):
            curr = curr_minutiae.locX, curr_minutiae.locY
            comp = comp_minutiae.locX, comp_minutiae.locY
            if math.dist(curr, comp) <= 4:
                count += 1
    for idx, curr_minutiae in enumerate(finger1[1]):
        for idx1, comp_minutiae in enumerate(finger2[1]):
            curr = curr_minutiae.locX, curr_minutiae.locY
            comp = comp_minutiae.locX, comp_minutiae.locY
            if math.dist(curr, comp) <= 4:
                count += 1
    return count
        
def train_compare(DA_DICT):
    close = []
    for key in DA_DICT.keys():
        if key[0] == "f":
            f = DA_DICT[key]
            s = DA_DICT[f's{key[1:]}']
            print(key[1:])
            close.append(count_close(f,s))
    return round(np.average(close))

def test_compare(min_close):
    comp_dict = read_in_image(TESTING_MIN, TESTING_MAX)
    for key in comp_dict.keys():
        if key[0] == "f":
            f = DA_DICT[key]
            s = DA_DICT[f's{key[1:]}']
            if count_close(f,s) >= min_close:
                print("pass")
            else:
                print("fail")
            
            

if __name__ == '__main__':
    DA_DICT = read_in_image(TRAINING_MIN, TRAINING_MAX)
    min_close = train_compare(DA_DICT)
