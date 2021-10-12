import cv2
import time
import numpy as np
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller

RECORDING = 1   # set this flag high if you want the screen recorded
DEBUG = 1       # set this flag high to enable debugging output

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

'''
Obstacle Templates

*** TO-DO ***
* Get images of the remaining obstacles
'''
TREE_ROOT1 = cv2.imread('./Obstacles/treeRoot1.png', 0)
TREE_ROOT2 = cv2.imread('./Obstacles/treeRoot2.png', 0)
TREE_ROOT3 = cv2.imread('./Obstacles/treeRoot3.png', 0)
TREE_ROOT4 = cv2.imread('./Obstacles/treeRoot4.png', 0)
TREE_TRUNK = cv2.imread('./Obstacles/treeSlide.png', 0)
GAP1 = cv2.imread('./Obstacles/gap1.png', 0)
GAP2 = cv2.imread('./Obstacles/gap2.png', 0)
FIRE_TRAP = cv2.imread('./Obstacles/fireTrap.png', 0)
LEFT_TURN = cv2.imread('./Obstacles/turnLeft.png', 0)
RIGHT_TURN = cv2.imread('./Obstacles/turnRight.png', 0)
CROSS1 = cv2.imread('./Obstacles/cross1.png', 0)
CROSS2 = cv2.imread('./Obstacles/cross2.png', 0)

OBSTACLES = [(TREE_ROOT1, 'treeRoot' ), 
             (TREE_ROOT2, 'treeRoot'),
             (TREE_ROOT3, 'treeRoot'),
             (TREE_ROOT4, 'treeRoot'),
             (TREE_TRUNK, 'treeTrunk'), 
             (GAP1, 'gap'),
             (GAP2, 'gap'),
             (FIRE_TRAP, 'fireTrap'),
             (LEFT_TURN, 'leftTurn'),
             (RIGHT_TURN, 'rightTurn'),
             (CROSS1, 'crossTurn'),
             (CROSS2, 'crossTurn')]

# OBSTACLES = [(LEFT_TURN, 'leftTurn'),
#              (RIGHT_TURN, 'rightTurn'),
#              (CROSS1, 'rightTurn')]

keyboard = Controller()

def display_template_match(frame, obstacle, maxLoc):
    startX, startY = maxLoc
    endX = startX + obstacle.shape[1]
    endY = startY + obstacle.shape[0]
    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)
    cv2.imshow('Template Matching', frame)

def check_for_obstacle(frame):
    for obstacle, group in OBSTACLES:
        result = cv2.matchTemplate(frame, obstacle, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        '''
        *** TO-DO ***
        * Gaps are not always detected if there are coins in the way
        '''
        if ((group == 'treeRoot' or group == 'gap' or group == 'fireTrap') and maxVal > 0.7):
            keyboard.press('w')
            time.sleep(0.025)
            keyboard.release('w')
            if DEBUG:
                display_template_match(frame, obstacle, maxLoc)
                # print(maxVal, group)
                # print()

        elif ((group == 'leftTurn' or group == 'rightTurn') and maxVal > 0.6):
            if group == 'leftTurn':
                keyboard.press('a')
                time.sleep(0.025)
                keyboard.release('a')
            else:
                keyboard.press('d')
                time.sleep(0.025)
                keyboard.release('d')
            if DEBUG:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()

        elif(group == 'crossTurn' and maxVal > 0.55):
            keyboard.press('d')
            time.sleep(0.025)
            keyboard.release('d')
            if DEBUG:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()

        elif(group == 'treeTrunk' and maxVal > 0.5):
            keyboard.press('s')
            time.sleep(0.025)
            keyboard.release('s')
            if DEBUG:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()


def main():
    # can vary the third parameter (FPS) to change the speed of the video
    output = cv2.VideoWriter('ScreenCaptures/temple_run_4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (540, 960), 0)

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))

            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

            check_for_obstacle(grayscale_frame[350:609, :])

            if RECORDING:
                output.write(grayscale_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                output.release()
                break

if __name__ == "__main__":
    main()