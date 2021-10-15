# -*- coding: utf-8 -*-
import cv2
from mss import mss
from PIL import Image
import numpy as np
import time
import os

from grabscreen import grab_screen
from getkeys import key_check

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 1000}

file_name = "D:/371Q_Project/TempleRun/Data/training_data.npy"
file_name2 = "D:/371Q_Project/TempleRun/Data/target_data.npy"


def get_data():

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        image_data = list(np.load(file_name, allow_pickle=True))
        targets = list(np.load(file_name2, allow_pickle=True))
    else:
        print('File does not exist, starting fresh!')
        image_data = []
        targets = []
    return image_data, targets


def save_data(image_data, targets):
    np.save(file_name, image_data)
    np.save(file_name2, targets)


def main():
    image_data, targets = get_data()
    count = 0

    while True:
        keys = key_check()
        print("waiting press B to start")
        if keys == "B":
            print("Starting")
            break
    
    with mss() as sct:
        while True:
            
            count +=1
            last_time = time.time()
            image = grab_screen(region=(dim['top'], dim['left'], dim['width'], dim['height']))
            #image = cv2.cvtbColor(image, cv2.COLOR_BGR2GRAY)
            #image = cv2.Canny(image, threshold1=119, threshold2=250)

            image = cv2.resize(image, (224, 224))

            # Debug line to show image
            #cv2.imshow("AI Peak", image)
            #cv2.waitKey(1)

            # Convert to numpy array
            image = np.array(image)
            image_data.append(image)

            keys = key_check()
            targets.append(keys)
            if keys == "H":
                break

            print('loop took {} seconds'.format(time.time()-last_time))
            
    save_data(image_data, targets)


if __name__ == "__main__":
    main()