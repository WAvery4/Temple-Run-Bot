# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:49:38 2021

@author: lwing
"""

import random
import time

from getkeys import key_check
import pydirectinput
import keyboard
import time
import cv2
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, W, A, S, D, J, L, I
from fastai.vision.all import *
import numpy as np

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def label_func(x): return x.parent.name
learn_inf = load_learner("D:/371Q_Project/TempleRun/5gammaNEW.pkl")
print("loaded learner")

# Sleep time after actions
sleepy = 0.2blwasllllllh

# Wait for me to push B to start.
print("waiting press B to start")
keyboard.wait('B')
time.sleep(sleepy)

labels = ['Jump', 'JumpLeanLeft', 'JumpLeanRight', 'LeanLeft', 
          'LeanRight', 'Left', 'Nothing', 'Right', 'Slide', 
          'SlideLeanLeft', 'SlideLeanRight']

# Hold down W no matter what!
#keyboard.press('I')

# Randomly pick action then sleep.
# 0 do nothing relbhease everything ( except W )
# 1 hold left
# 2 hold right
# 3 Press Jump

while True:

    image = grab_screen(region=(0, 0, 540, 600))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (128, 128))
    # cv2.imshow("Fall", image)
    # cv2.waitKey(1)
    start_time = time.time()
    result = learn_inf.predict(image)
    index = np.argmax(result[1])
    index = index.numpy()
    #print(index)
    #action = result[0]
    #print(action[0])
    print(result[0])
    print(result[1])
    action = labels[index]
    #print(action)
    #action = "Jump"

    
    if action == "Jump":
        print(f"JUMP!")
        keyboard.press("W")
        keyboard.release("A")
        keyboard.release("S")
        keyboard.release("D")
        keyboard.release("J")
        keyboard.release("L")
        time.sleep(sleepy)
        keyboard.release("W")

    if action == "Nothing":
        print("Doing nothing....")
        keyboard.release("W")
        keyboard.release("A")
        keyboard.release("S")
        keyboard.release("D")
        keyboard.release("J")
        keyboard.release("L")
        time.sleep(sleepy)

    elif action == "Left":
        print(f"LEFT!")
        keyboard.release("W")
        keyboard.press("A")
        keyboard.release("S")
        keyboard.release("D")
        keyboard.release("J")
        keyboard.release("L")
        time.sleep(sleepy)
        keyboard.release("A")
    
    elif action == "Right":
        print(f"Right!")
        keyboard.release("W")
        keyboard.release("A")
        keyboard.release("S")
        keyboard.press("D")
        keyboard.release("J")
        keyboard.release("L")
        time.sleep(sleepy)
        keyboard.release("D")
        
    elif action == "Slide":
        print(f"Slide!")
        keyboard.release("W")
        keyboard.release("A")
        keyboard.press("S")
        keyboard.release("D")
        keyboard.release("J")
        keyboard.release("L")
        time.sleep(sleepy)
        keyboard.release("S")
        
    elif action == "LeanLeft":
        print(f"Lean Left!")
        keyboard.release("W")
        keyboard.release("A")
        keyboard.release("S")
        keyboard.release("D")
        keyboard.press("J")
        keyboard.release("L")
        time.sleep(sleepy)
        keyboard.release("J")
        
    elif action == "LeanRight":
        print(f"Lean Right!")
        keyboard.release("W")
        keyboard.release("A")
        keyboard.release("S")
        keyboard.release("D")
        keyboard.release("J")
        keyboard.press("L")
        time.sleep(sleepy)
        keyboard.release("L")
        
    elif action == "JumpLeanLeft":
        print(f"Jump Lean Left!")
        keyboard.press("W")
        keyboard.release("A")
        keyboard.release("S")
        keyboard.release("D")
        keyboard.press("J")
        keyboard.release("L")
        time.sleep(sleepy)
        keyboard.release("J")
        keyboard.release("W")
        
    elif action == "JumpLeanRight":
        print(f"Jump Lean Right!")
        keyboard.press("W")
        keyboard.release("A")
        keyboard.release("S")
        keyboard.release("D")
        keyboard.release("J")
        keyboard.press("L")
        time.sleep(sleepy)
        keyboard.release("L")
        keyboard.release("W")
        
    elif action == "SlideLeanLeft":
        print(f"Slide Lean Left!")
        keyboard.release("W")
        keyboard.release("A")
        keyboard.press("S")
        keyboard.release("D")
        keyboard.press("J")
        keyboard.release("L")
        time.sleep(sleepy)
        keyboard.release("J")
        keyboard.release("S")
        
        
    elif action == "SlideLeanRight":
        print(f"Slide Lean Right!")
        keyboard.release("W")
        keyboard.release("A")
        keyboard.press("S")
        keyboard.release("D")
        keyboard.release("J")
        keyboard.press("L")
        time.sleep(sleepy)
        keyboard.release("L")
        keyboard.release("S")
        


    # End simulation by hitting h
    keys = key_check()
    if keys == "H":
        break

keyboard.release('I')