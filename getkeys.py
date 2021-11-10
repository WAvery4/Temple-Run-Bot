# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:04:22 2021

@author: lwing
"""

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    if 'H' in keys:
        return 'H'
    elif 'S' in keys and 'J' in keys:
        return 'SJ'
    elif 'S' in keys and 'L' in keys:
        return 'SL'
    elif 'W' in keys and 'J' in keys:
        return 'WJ'
    elif 'W' in keys and 'L' in keys:
        return 'WL'
    elif 'B' in keys:
        return 'B'
    elif 'A' in keys:
        return 'A'
    elif 'D' in keys:
        return 'D'
    elif 'W' in keys:
        return 'W'
    elif 'S' in keys:
        return 'S'
    elif 'J' in keys:
        return 'J'
    elif 'K' in keys:
        return 'K'
    elif 'L' in keys:
        return 'L'
    else:
        return 'I'