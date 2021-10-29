import cv2
from mss import mss
from PIL import Image
import numpy as np
from time import time, sleep
from pynput.keyboard import Controller, Listener, Key

DEBUG = 1              # set this flag high to enable debugging in the main loop

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

kb = Controller()

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
STONE_GAP = cv2.imread('./Obstacles/stone_gap.jpg', 0)
FIRE = cv2.imread('./Obstacles/fire.jpg', 0)
STONE_GATE = cv2.imread('./Obstacles/stone_gate.jpg', 0)

OBSTACLES = [(TREE_ROOT1, 'treeRoot' ),
             (TREE_ROOT2, 'treeRoot'),
             (TREE_ROOT3, 'treeRoot'),
             (TREE_ROOT4, 'treeRoot'),
             (TREE_TRUNK, 'treeTrunk'),
             (GAP1, 'gap'),
             (GAP2, 'gap'),
             (STONE_GAP, 'gap'),
             (FIRE, 'fire'),
             (STONE_GATE, 'stoneGate')]


def display_template_match(frame, obstacle, maxLoc):
    """
    Displays the bounding box for a template match.
    """
    startX, startY = maxLoc
    endX = startX + obstacle.shape[1]
    endY = startY + obstacle.shape[0]
    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)


def check_for_obstacle(frame, debug=0):
    """
    Loops through the obstacles associated with the current state and makes an action
    if that obstacle is detected.
    """
    for obstacle, group in OBSTACLES:
        result = cv2.matchTemplate(frame, obstacle, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        if (group == 'treeRoot' or group == 'gap' or group == 'fire') and maxVal > 0.7:
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()

        if (group == 'stoneGate') and maxVal > 0.3:
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()

        elif group == 'treeTrunk' and maxVal > 0.55:
            kb.press('s')
            sleep(0.025)
            kb.release('s')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()

        if debug:
            cv2.imshow('Template Matching', frame)


def check_for_turn(frame):
    """
    Averages the pixels of three patches located on the screen and performs
    a turn based on the results.
    """
    patch0 = frame[100:150, 50:100]
    patch1 = frame[50:100, 250:300]
    patch2 = frame[100:150, 440:490]
    patch0_average = np.average(patch0)
    patch1_average = np.average(patch1)
    patch2_average = np.average(patch2)
    if patch1_average < 80:
        if patch0_average > 80:
            kb.press('a')
            sleep(0.025)
            kb.release('a')
        elif patch2_average > 80:
            kb.press('d')
            sleep(0.025)
            kb.release('d')


def on_press(key):
    pass


def on_release(key):
    """
    Saves the current frame when the space bar is pressed
    """
    if key == Key.space:
        print("Screen shot")
        with mss() as sct:
            frame = np.array(sct.grab(dim))
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite('./Screenshots/img' + str(time()) + '.jpg', image_gray)


def main():

    if DEBUG:
        loop_time = time()

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # focus on a 260x640 region of the frame
            obstacle_region = image_gray[350:610, :]
            template_gray = FIRE

            # template matching
            check_for_obstacle(obstacle_region, debug=1)

            # averaging patch histograms
            check_for_turn(obstacle_region)

            # debug the loop rate
            # if DEBUG:
            #     print('FPS {}'.format(1 / (time() - loop_time)))
            #     loop_time = time()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    main()
