import cv2
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from mss.windows import MSS as mss
from pynput.keyboard import Controller, Listener, Key, KeyCode
import threading

RECORDING = 0  # set this flag high if you want the screen recorded
DEBUG = 0  # set this flag high to enable debugging in the main loop

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
TREE_ROOT5 = cv2.imread('./Obstacles/Temple/treeRoot5.png', 0)
TREE_TRUNK = cv2.imread('./Obstacles/Temple/treeSlide.png', 0)
GAP1 = cv2.imread('./Obstacles/Temple/gap1.png', 0)
GAP2 = cv2.imread('./Obstacles/Temple/gap2.png', 0)
FIRE_TRAP = cv2.imread('./Obstacles/Temple/fireTrap.png', 0)
ROCK_LEVEL = cv2.imread('./Obstacles/Temple/rockLevel.png', 0)
ROCK_LEVEL2 = cv2.imread('./Obstacles/Temple/rockLevel2.png', 0)
WATER_LEVEL = cv2.imread('./Obstacles/Temple/waterLevel.png', 0)
TURN = cv2.imread('./Obstacles/Temple/turn_template.png', 0)

'''
Rock Obstacle Templates
*** TO-DO ***
* Get images of the remaining obstacles
'''
STONE_TREE = cv2.imread('./Obstacles/Water/stoneTree.png', 0)

'''
Water Obstacle Templates
*** TO-DO ***
* Get images of the remaining obstacles
'''
TIKI = cv2.imread('./Obstacles/Water/tiki.png', 0)
WATER_GAP = cv2.imread('./Obstacles/Water/waterGap.png', 0)
TEMPLE_LEVEL = cv2.imread('./Obstacles/Water/templeLevel.png', 0)

TEMPLE_OBSTACLES = [(TREE_ROOT1, 'treeRoot'),
                    (TREE_ROOT2, 'treeRoot'),
                    (TREE_ROOT3, 'treeRoot'),
                    (TREE_ROOT4, 'treeRoot'),
                    (TREE_ROOT5, 'treeRoot'),
                    (TREE_TRUNK, 'treeTrunk'),
                    (GAP1, 'gap'),
                    (GAP2, 'gap'),
                    (FIRE_TRAP, 'fireTrap'),
                    (ROCK_LEVEL, 'rockLevel'),
                    (ROCK_LEVEL2, 'rockLevel'),
                    (WATER_LEVEL, 'waterLevel'),
                    (TURN, 'turn')]

ALTERNATE_OBSTACLES = [(TIKI, 'tiki'),
                       (WATER_GAP, 'waterGap'),
                       (STONE_TREE, 'stoneTree'),
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

        if (group == 'treeRoot') and maxVal > 0.75:
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()
            return

        elif (group == 'gap') and maxVal > 0.7:
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()
            return

        elif (group == 'fireTrap') and maxVal > 0.7:
            kb.press('s')
            sleep(0.025)
            kb.release('s')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()
            return

        elif (group == 'treeTrunk' or group == 'stoneTree') and maxVal > 0.55:
            kb.press('s')
            sleep(0.025)
            kb.release('s')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()
            return

        # elif group == 'turn' and maxVal > 0.4:
        #     kb.press('a')
        #     sleep(0.025)
        #     kb.release('a')
        #     if debug:
        #         display_template_match(frame, obstacle, maxLoc)
        #         print(maxVal, group)
        #         print()
        #     return

        elif (group == 'waterGap' or group == 'tiki') and maxVal > 0.45:
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
                print()
            return

        elif group == 'rockLevel' and maxVal > 0.5:
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            OBSTACLES = ALTERNATE_OBSTACLES
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)
            return

        # elif group == 'waterLevel' and maxVal > 0.5:
        #     OBSTACLES = ALTERNATE_OBSTACLES
        #     if debug:
        #         display_template_match(frame, obstacle, maxLoc)
        #         print(maxVal, group)

        elif group == 'templeLevel' and maxVal > 0.5:
            kb.press('w')
            sleep(0.025)
            kb.release('w')
            OBSTACLES = TEMPLE_OBSTACLES
            if debug:
                display_template_match(frame, obstacle, maxLoc)
                print(maxVal, group)

    #check_for_turn(frame)


def check_for_turn(frame):
    '''
    Averages the pixels of three patches located on the screen and performs
    a turn based on the results.
    '''
    th, binary = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
    cv2.rectangle(binary, (50, 150), (100, 200), (255, 0, 0), 3)
    cv2.rectangle(binary, (250, 50), (300, 100), (255, 0, 0), 3)
    cv2.rectangle(binary, (440, 150), (490, 200), (255, 0, 0), 3)
    cv2.imshow('Turn', binary)

    patch0 = binary[150:200, 50:100]
    patch1 = binary[50:100, 250:300]
    patch2 = binary[150:200, 440:490]
    patch0_average = np.average(patch0)
    patch1_average = np.average(patch1)
    patch2_average = np.average(patch2)
    # if patch1_average < 90 and (patch0_average > 100 or patch2_average > 100):
    # print(patch0_average, patch1_average, patch2_average, 'turn')
    if patch0_average > 100 and patch1_average < 100:
        print(patch0_average, patch1_average, 'turn left')
        kb.press('a')
        sleep(0.025)
        kb.release('a')
    elif patch2_average > 100 and patch1_average < 100:
        print(patch2_average, patch1_average, 'turn right')
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
            obstacle_region = grayscale_frame[400:660, :]

            # template matching
            check_for_obstacle(obstacle_region, debug=1)

            # averaging patch histograms
            # check_for_turn(obstacle_region)

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

def turn():
    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # focus on a 260x640 region of the frame
            obstacle_region = grayscale_frame[375:635, :]

            # averaging patch histograms
            check_for_turn(obstacle_region)

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
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite('./Screenshots/img' + str(time()) + '.PNG', image_gray)
    elif key == KeyCode.from_char('a'):
        print("Pressed a")
    elif key == KeyCode.from_char('d'):
        print("Pressed d")
    elif key == KeyCode.from_char('w'):
        print("Pressed w")
    elif key == KeyCode.from_char('s'):
        print("Pressed s")


def on_release(key):
    pass


# def main():
#     while(True):
#         pass


if __name__ == "__main__":
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    main()

    # t1 = threading.Thread(target=obstacles)
    # t2 = threading.Thread(target=turn)
    # t1.start()
    # t2.start()
    #
    # t1.join()
    # t2.join()