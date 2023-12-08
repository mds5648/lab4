import cv2
import os
import fingerprint_feature_extractor


TRAINING_MAX = 5 #1500
TRAINING_MIN = 1 #0

TESTING_MAX = 7 #2000
TESTING_MIN = 6 #1501

PATH_TO_IMAGES = "./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt"


def read_in_image(min, max):
    images_dic = {}
    for count in range (min, max + 1):
        folder = PATH_TO_IMAGES + "/figs_" + str(count)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True)
                images_dic.update({filename : (FeaturesBifurcations, FeaturesTerminations)})
                print(FeaturesBifurcations)
                print(FeaturesTerminations)
    return images_dic
                

read_in_image(1, 3)