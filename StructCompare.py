import os
import cv2
from multiprocessing import Pool, Manager
import math
import random
import numpy as np
from skimage.metrics import structural_similarity

TRAINING_MAX = 5 #1500
TRAINING_MIN = 0 #0

TESTING_MAX = 7 #2000
TESTING_MIN = 6 #1501

PATH_TO_IMAGES = "./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt"

def read_in_image(min, max):
    with Manager() as manager:
        images_dic = manager.dict()
        lock = manager.Lock()
        with Pool() as pool:
            args = []
            for count in range(min, max + 1):
                folder = PATH_TO_IMAGES + "/figs_" + str(count)
                for filename in os.listdir(folder):
                    path = f"{folder}/{filename}"
                    args.append((path, images_dic, lock))
            pool.map(process_image, args)
        return dict(images_dic)
                    
def process_image(args):
    filename, images_dic, lock = args
    img = cv2.imread(filename, 0)
    if img is not None:
        print(filename.split("/")[-1])
        with lock:
            images_dic[filename.split("/")[-1]] = img
            
def split_dict(images_dic):
    s = {}
    f = {}
    for key in list(images_dic.keys()):
        if key[0] == "f":
            f[key] = images_dic[key]
        else:
            s[key] = images_dic[key]
    return f,s

def percent_match(finger1, finger2, distance):
    return structural_similarity(finger1,finger2)* 100

def train_compare(f, s, dist, tolerance):
    close = []
    notClose = []
    s2 = None
    first = True
    for key in f.keys():
        fFinger = f[key]
        sFinger = s[f's{key[1:]}']
        if not first:
            notClose.append(percent_match(fFinger,s2,dist))
        first = False
        s2 = sFinger
        close.append(percent_match(fFinger,sFinger,dist))
    match, nonMatch = np.average(close), np.average(notClose)
    return match-((match-nonMatch)*tolerance*2) #Why 2? with tolerances from 0 to 1, multiplying by 3 makes the FAR and FRR cross paths usually

def FRR(f,s,min_match, dist):
    pAss, fAil = 0,0
    for key in f.keys():
        fFinger = f[key]
        sFinger = s[f's{key[1:]}']
        if percent_match(sFinger,fFinger,dist) >= min_match:
            pAss += 1
        else:
            fAil += 1
    print(f'FRR = {fAil/(pAss+fAil)*100}%')
    
def FAR(f,s,min_match, dist):
    pAss, fAil = 0,0
    for key in f.keys():
        fFinger = f[key]
        rando_s = []
        
        while len(rando_s) < 5:
            choice = random.choice(list(s.keys()))
            if choice[1:] != key[1:]:
                rando_s.append(choice)
        for choice in rando_s:
            sFinger = s[choice]
            if percent_match(fFinger,sFinger, dist) >= min_match:
                fAil += 1
            else: 
                pAss += 1
    print(f'FAR = {fAil/(pAss+fAil)*100}%')
    
def test_compare(f,s,min_match, dist):
    FRR(f,s,min_match,dist)
    FAR(f,s,min_match,dist)
    
def test(train,test,tolerance,dist=10):
    f1,s1 = split_dict(train)
    f2,s2 = split_dict(test)
    min_match = train_compare(f1,s1,dist,tolerance)
    print("Tolerance = " + str(tolerance))
    test_compare(f2,s2, min_match, dist)
    
if __name__ == '__main__':
    train = read_in_image(TRAINING_MIN, TRAINING_MAX)
    testt = read_in_image(TESTING_MIN, TESTING_MAX)
    for tol in range (0,11):
        test(train,testt,tol/10,dist=50)