import cv2
from mss import mss
from PIL import Image
import numpy as np
from time import time, sleep
from pynput.keyboard import Controller, Listener

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

kb = Controller()

FIRE = cv2.imread('./Obstacles/fire.jpg', 0)

def main():

    found_fire = False

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))[:,:,:3]

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # focus on a 260x640 region of the frame
            obstacle_region = image_gray[350:610, :]
            template_gray = FIRE

            result = cv2.matchTemplate(obstacle_region, template_gray, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

            if maxVal > 0.7:
                # determine the starting and ending (x, y)-coordinates of the bounding box
                (startX, startY) = maxLoc
                endX = startX + template_gray.shape[1]
                endY = startY + template_gray.shape[0]
                # if not found_fire:
                #     cv2.imwrite('./Screenshots/fire1.jpg', obstacle_region)
                #     found_fire = True

                # jump over fire
                kb.press('w')
                sleep(0.025)
                kb.release('w')

                # draw the bounding box on the image
                cv2.rectangle(obstacle_region, (startX, startY), (endX, endY), (255, 0, 0), 3)

            cv2.imshow('detect', obstacle_region)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()