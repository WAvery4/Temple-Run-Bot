import cv2
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from mss.windows import MSS as mss
from pynput.keyboard import Controller, Listener, Key, KeyCode

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

def main():

    with mss() as sct:
        num = 0
        while True:
            frame = np.array(sct.grab(dim))
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # focus on a 260x640 region of the frame
            obstacle_region = grayscale_frame[400:660, :]

            # cv2.imwrite('./Screenshots/img' + str(num) + '.PNG', obstacle_region)
            # num += 1

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break


def on_press(key):
    """
    Saves the current frame when the space bar is pressed
    """
    if key == Key.space:
        print("Screen shot")
        with mss() as sct:
            frame = np.array(sct.grab(dim))
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # focus on a 260x640 region of the frame
            obstacle_region = grayscale_frame[400:660, :]
            cv2.imwrite('./Screenshots/img' + str(time()) + '.PNG', obstacle_region)


def on_release(key):
    pass


if __name__ == "__main__":
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    main()