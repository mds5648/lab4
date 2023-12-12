import MinutiaeExtract
import orbCompare
import StructCompare

import numpy as np
import math
import random

TRAINING_MAX = 5 #1500
TRAINING_MIN = 0 #0

TESTING_MAX = 7 #2000
TESTING_MIN = 6 #1501

PATH_TO_IMAGES = "./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt"

def hybridCheck(fingers,dists,mins):
    finger1, finger2, min1, min2 = fingers
    distM, distO, distS = dists
    minM, minO, minS = mins
    passing = 0
    
    if orbCompare.percent_match(finger1,finger2,distO) >= minO:
        passing+=1
    if StructCompare.percent_match(finger1,finger2,distS) >= minS:
        passing+=1
    if MinutiaeExtract.count_close(min1,min2,distM) >= minM:
        passing+=1
        
    if passing >=2:
        return True
    return False

def FRR(f,s,da_dict, dists, mins):
    pAss, fAil = 0,0
    for key in f.keys():
        finger1 = f[key]
        finger2 = s[f's{key[1:]}']
        min1 = da_dict[key]
        min2 = da_dict[f's{key[1:]}']
        
        fingers = (finger1, finger2, min1, min2)
        if hybridCheck(fingers,dists,mins):
            pAss += 1
        else:
            fAil += 1
    print(f'FRR = {fAil/(pAss+fAil)*100}%')
    
def FAR(f,s,da_dict, dists, mins):
    pAss, fAil = 0,0
    for key in f.keys():
        finger1 = f[key]
        min1 = da_dict[key]
        rando_s = []
        
        while len(rando_s) < 5:
            choice = random.choice(list(s.keys()))
            if choice[1:] != key[1:]:
                rando_s.append(choice)
        for choice in rando_s:
            finger2 = s[choice]
            min2 = da_dict[choice]
            fingers = (finger1, finger2, min1, min2)
            if hybridCheck(fingers,dists,mins):
                fAil += 1
            else:
                pAss += 1
    print(f'FAR = {fAil/(pAss+fAil)*100}%')
            
def test_compare(f,s,da_dict,dists,mins):
    FRR(f,s,da_dict,dists,mins)
    FAR(f,s,da_dict,dists,mins)
    
def test(fs,ss,da_dicts, tol):
    f1,f2 = fs
    s1,s2 = ss
    da_dict, comp_dict = da_dicts
    
    dists = (15, 50, 50)
    
    minM = MinutiaeExtract.train_compare(da_dict, dists[0], tol/10)
    minO = orbCompare.train_compare(f1,s1,dists[1], tol/10)
    minS = StructCompare.train_compare(f1,s1,dists[2], tol/10)
    
    mins = (minM, minO, minS)
    
    print("Tolerance = " + str(tol))
    test_compare(f2,s2,comp_dict,dists,mins)
    
    
    

if __name__ == "__main__":
    train = orbCompare.read_in_image(TRAINING_MIN, TRAINING_MAX)
    testt = orbCompare.read_in_image(TESTING_MIN, TESTING_MAX)
    da_dict = MinutiaeExtract.read_in_image(TRAINING_MIN, TRAINING_MAX)
    comp_dict = MinutiaeExtract.read_in_image(TESTING_MIN, TESTING_MAX)
    
    f,s = orbCompare.split_dict(train)
    f_t, s_t = orbCompare.split_dict(testt)
    
    fs = f,f_t
    ss = s,s_t
    da_dicts = da_dict, comp_dict
    
    for tol in range(0,11):
        test(fs,ss,da_dicts, tol/10)