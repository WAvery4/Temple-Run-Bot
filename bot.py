import cv2
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from mss.windows import MSS as mss
from pynput.keyboard import Controller, Listener

RECORDING = 1          # set this flag high if you want the screen recorded
DEBUG = 0              # set this flag high to enable debugging in the main loop

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

'''
Temple Obstacle Templates

*** TO-DO ***
* Get images of the remaining obstacles
'''
TREE_ROOT1 = cv2.imread('./Obstacles/Temple/treeRoot1.png', 0)
TREE_ROOT2 = cv2.imread('./Obstacles/Temple/treeRoot2.png', 0)
TREE_ROOT3 = cv2.imread('./Obstacles/Temple/treeRoot3.png', 0)
TREE_ROOT4 = cv2.imread('./Obstacles/Temple/treeRoot4.png', 0)
TREE_TRUNK = cv2.imread('./Obstacles/Temple/treeSlide.png', 0)
GAP1 = cv2.imread('./Obstacles/Temple/gap1.png', 0)
GAP2 = cv2.imread('./Obstacles/Temple/gap2.png', 0)
FIRE_TRAP = cv2.imread('./Obstacles/Temple/fireTrap.png', 0)
ROCK_LEVEL = cv2.imread('./Obstacles/Temple/rockLevel.png', 0)
ROCK_LEVEL2 = cv2.imread('./Obstacles/Temple/rockLevel2.png', 0)
ALTERNATE_LEVEL = cv2.imread('./Obstacles/Temple/waterLevel.png', 0)

'''
Rock Obstacle Templates

*** TO-DO ***
* Get images of the remaining obstacles
'''

'''
Water Obstacle Templates

*** TO-DO ***
* Get images of the remaining obstacles
'''
TIKI = cv2.imread('./Obstacles/Water/tiki.png', 0)
WATER_GAP = cv2.imread('./Obstacles/Water/waterGap.png', 0)
TEMPLE_LEVEL = cv2.imread('./Obstacles/Water/templeLevel.png', 0)

TEMPLE_OBSTACLES = [(TREE_ROOT1, 'treeRoot' ), 
                    (TREE_ROOT2, 'treeRoot'),
                    (TREE_ROOT3, 'treeRoot'),
                    (TREE_ROOT4, 'treeRoot'),
                    (TREE_TRUNK, 'treeTrunk'), 
                    (GAP1, 'gap'),
                    (GAP2, 'gap'),
                    (FIRE_TRAP, 'fireTrap'),
                    (ALTERNATE_LEVEL, 'alternateLevel')]

ALTERNATE_OBSTACLES = [(TIKI, 'tiki'),
                       (WATER_GAP, 'waterGap'),
                       (TEMPLE_LEVEL, 'templeLevel')]

OBSTACLES = TEMPLE_OBSTACLES

kb = Controller()

def display_template_match(frame, obstacle, maxLoc):
    '''
    Displays the bounding box for a template match.
    '''
    startX, startY = maxLoc
    endX = startX + obstacle.shape[1]
    endY = startY + obstacle.shape[0]
    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)
    cv2.imshow('Template Matching', frame)

def check_for_obstacle(frame, debug=0):
    '''
    Loops through the obstacles associated with the current state and makes an action
    if that obstacle is detected.
    '''
    global OBSTACLES
    global TEMPLE_OBSTACLES
    global ALTERNATE_OBSTACLES

    for obstacle, group in OBSTACLES:
        result = cv2.matchTemplate(frame, obstacle, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        if ((group == 'treeRoot' or group == 'gap' or group == 'fireTrap') and maxVal > 0.7):
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()

        elif (group == 'treeTrunk' and maxVal > 0.55):
            kb.press('s')
            sleep(0.025)
            kb.release('s')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()

        elif (group == 'alternateLevel' and maxVal > 0.5):
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            OBSTACLES = ALTERNATE_OBSTACLES
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)

        elif (group == 'templeLevel' and maxVal > 0.5):
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            OBSTACLES = TEMPLE_OBSTACLES
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)


def check_for_turn(frame):
    '''
    Averages the pixels of three patches located on the screen and performs
    a turn based on the results.
    '''
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

def main():
    if RECORDING:
        output = cv2.VideoWriter('ScreenCaptures/temple_run_4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (540, 260), 0)

    if DEBUG:
        loop_time = time()
    
    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # focus on a 260x640 region of the frame
            obstacle_region = grayscale_frame[350:610,:] 
            
            # template matching
            check_for_obstacle(obstacle_region, debug=1)

            # averaging patch histograms
            check_for_turn(obstacle_region)

            if RECORDING:
                output.write(obstacle_region)

            # debug the loop rate
            if DEBUG:
                print('FPS {}'.format(1 / (time() - loop_time)))
                loop_time = time()

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                if RECORDING:
                    output.release()
                break

if __name__ == "__main__":
    main()