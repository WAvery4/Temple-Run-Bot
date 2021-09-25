import cv2
from mss import mss
from PIL import Image
import numpy as np

dim = {'top': 0, 'left': 0, 'width': 1000, 'height': 1000}

def main():
    with mss() as sct:
        while True:
            img = np.array(sct.grab(dim))
            cv2.imshow('test', img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()