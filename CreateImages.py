# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:48:16 2021

@author: lwing
"""

import cv2
import numpy as np

data = np.load("D:/371Q_Project/TempleRun/DataNew/training_datanew.npy", allow_pickle=True)
targets = np.load("D:/371Q_Project/TempleRun/DataNew/target_datanew.npy", allow_pickle=True)

print(f'Image Data Shape: {data.shape}')
print(f'targets Shape: {targets.shape}')

# Lets see how many of each type of move we have.
unique_elements, counts = np.unique(targets, return_counts=True)
print(np.asarray((unique_elements, counts)))

# Store both data and targets in a list.
# We may want to shuffle down the road.

holder_list = []
for i, image in enumerate(data):
    holder_list.append([data[i], targets[i]])

count_jump = 0
count_left = 0
count_right = 0
count_slide = 0
count_leanright = 0
count_leanleft = 0
count_nothing = 0
count_jumpleanright = 0
count_jumpleanleft = 0
count_slideleanright = 0
count_slideleanleft = 0

for data in holder_list:
    #print(data[1])
    if data[1] == 'W':
        count_jump += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/Jump/H7-u{count_jump}.png", data[0]) 
    elif data[1] == 'A':
        count_left += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/Left/H7-l{count_left}.png", data[0]) 
    elif data[1] == 'D':
        count_right += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/Right/H7-r{count_right}.png", data[0]) 
    elif data[1] == 'S':
        count_slide += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/Slide/H7-j{count_slide}.png", data[0])
    elif data[1] == 'I':
        count_nothing += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/Nothing/H7-u{count_nothing}.png", data[0]) 
    elif data[1] == 'J':
        count_leanleft += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/LeanLeft/H7-l{count_leanleft}.png", data[0]) 
    elif data[1] == 'L':
        count_leanright += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/LeanRight/H7-r{count_leanright}.png", data[0])
    elif data[1] == 'WJ':
        count_jumpleanleft += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/JumpLeanLeft/H7-l{count_jumpleanleft}.png", data[0]) 
    elif data[1] == 'WL':
        count_jumpleanright += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/JumpLeanRight/H7-r{count_jumpleanright}.png", data[0])
    elif data[1] == 'SJ':
        count_slideleanleft += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/SlideLeanLeft/H7-l{count_slideleanleft}.png", data[0]) 
    elif data[1] == 'SL':
        count_slideleanright += 1
        cv2.imwrite(f"D:/371Q_Project/TempleRun/DataNew/SlideLeanRight/H7-r{count_slideleanright}.png", data[0])