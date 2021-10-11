import cv2
import time
import numpy as np
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller

RECORDING = 0   # set this flag high if you want the screen recorded
DEBUG = 1       # set this flag high to enable debugging output

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

'''
Obstacle Templates

*** TO-DO ***
* Get images of the remaining obstacles
'''
TREE_ROOT1 = cv2.imread('./Obstacles/treeRoot1.jpg', 0)
TREE_ROOT2 = cv2.imread('./Obstacles/treeRoot2.jpg', 0)
TREE_ROOT3 = cv2.imread('./Obstacles/treeRoot3.jpg', 0)
TREE_ROOT4 = cv2.imread('./Obstacles/treeRoot4.jpg', 0)
GAP1 = cv2.imread('./Obstacles/gap1.jpg', 0)
GAP2 = cv2.imread('./Obstacles/gap2.jpg', 0)
RIGHT = cv2.imread('./Obstacles/right.jpg', 0)

OBSTACLES = [(TREE_ROOT1, 'treeRoot' ), 
             (TREE_ROOT2, 'treeRoot'),
             (TREE_ROOT3, 'treeRoot'),
             (TREE_ROOT4, 'treeRoot'), 
             (GAP1, 'gap'), 
             (GAP2, 'gap'),
             (RIGHT, 'rightTurn')]

# OBSTACLES = [(GAP1, 'gap'), 
#              (GAP2, 'gap')]

keyboard = Controller()

def check_for_obstacle(frame):
    for obstacle, group in OBSTACLES:
        result = cv2.matchTemplate(frame, obstacle, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        '''
        *** TO-DO ***
        * Resolve double jump issues with TREE_ROOT3/4
        * Resolve issues where gaps are sometimes undetected (something other than template matching?)
        '''
        if (group == 'treeRoot' and maxVal > 0.7) or (group == 'gap' and maxVal > 0.5):
            keyboard.press('w')
            time.sleep(0.05)
            keyboard.release('w')
            if DEBUG:
                startX, startY = maxLoc
                endX = startX + obstacle.shape[1]
                endY = startY + obstacle.shape[0]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)
                cv2.imshow('Template Matching', frame)
                print(maxVal)

        '''
        *** TO-DO ***
        * Implement logic for turning left/right based on template match.
        '''
        # elif (group == 'rightTurn' and maxVal > 0.7):
        #     keyboard.press('d')
        #     time.sleep(0.05)
        #     keyboard.release('d')
        #     if DEBUG:
        #         startX, startY = maxLoc
        #         endX = startX + obstacle.shape[1]
        #         endY = startY + obstacle.shape[0]
        #         cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)
        #         cv2.imshow('Template Matching', frame)
        #         print(maxVal)

        


def main():
    # can vary the third parameter (FPS) to change the speed of the video
    output = cv2.VideoWriter('ScreenCaptures/temple_run_4.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (540, 960))

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))[:,:,:3]
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            check_for_obstacle(grayscale_frame)

            if RECORDING:
                output.write(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                output.release()
                break

if __name__ == "__main__":
    main()