import cv2
from time import time, sleep
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mss.windows import MSS as mss
from pynput.keyboard import Controller, Listener

RECORDING = 0          # set this flag high if you want the screen recorded
DEBUG = 0              # set this flag high to enable debugging in the main loop

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 600}

def unpickle_desc(path):
    '''
    Description:
        Unpickles a pickled SIFT descriptor object
    Input:
        - path (str): File path to a pickled descriptor
    Output:
        - A descriptor object used for feature matching
    '''
    file = open(path,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file

'''
Temple Obstacle Templates
'''
TREE_ROOT1 = cv2.imread('./Obstacles/Temple/treeRoot1_2.png', 0)
TREE_ROOT2 = cv2.imread('./Obstacles/Temple/treeRoot2_1.png', 0)
TREE_ROOT3 = cv2.imread('./Obstacles/Temple/treeRoot3.png', 0)
TREE_ROOT4 = cv2.imread('./Obstacles/Temple/treeRoot4.png', 0)
TREE_TRUNK = unpickle_desc('./Obstacles/Temple/ORB/treeSlide1.pickle')
GAP1 = unpickle_desc('./Obstacles/Temple/ORB/gap1.pickle')
GAP2 = unpickle_desc('./Obstacles/Temple/ORB/gap2.pickle')
FIRE_TRAP = unpickle_desc('./Obstacles/Temple/ORB/fireTrap.pickle')
ROCK_LEVEL = cv2.imread('./Obstacles/Temple/rockLevel.png', 0)
ROCK_LEVEL2 = cv2.imread('./Obstacles/Temple/rockLevel2.png', 0)
ALTERNATE_LEVEL = unpickle_desc('./Obstacles/Temple/ORB/alternateLevel.pickle')

TREE_ROOT1_NAME = 'treeRoot1'
TREE_ROOT2_NAME = 'treeRoot2'
TREE_ROOT3_NAME = 'treeRoot3'
TREE_ROOT4_NAME = 'treeRoot4'
TREE_TRUNK_NAME = 'treeTrunk'
GAP1_NAME = 'gap1'
GAP2_NAME = 'gap2'
FIRE_TRAP_NAME = 'fireTrap'
ALTERNATE_LEVEL_NAME = 'alternateLevel'

'''
Rock Obstacle Templates
'''
STONE_TREE_TRUNK = cv2.imread('./Obstacles/Water/stoneTreeSlide2.png', 0)
STONE_TREE_TRUNK1 = cv2.imread('./Obstacles/Water/stoneTreeSlide3.png', 0)
STONE_GAP1 = unpickle_desc('./Obstacles/Temple/ORB/stoneGap1.pickle')
STONE_GAP2 = unpickle_desc('./Obstacles/Temple/ORB/stoneGap2.pickle')
TEMPLE_LEVEL_STONE = unpickle_desc('./Obstacles/Temple/ORB/templeLevelStone.pickle')

STONE_TRUNK_NAME = 'treeTrunkStone'
STONE_TRUNK1_NAME = 'treeTrunkStone2'
STONE_GAP1_NAME = 'stoneGap1'
STONE_GAP2_NAME = 'stoneGap2'
TEMPLE_LEVEL_STONE_NAME = 'templeLevelStone'

'''
Water Obstacle Templates
'''
TIKI = cv2.imread('./Obstacles/Water/stoneGate2.png', 0)
WATER_GAP1 = unpickle_desc('./Obstacles/Temple/ORB/waterGap1.pickle')
WATER_GAP2 = unpickle_desc('./Obstacles/Temple/ORB/waterGap2.pickle')
TEMPLE_LEVEL_WATER = unpickle_desc('./Obstacles/Temple/ORB/templeLevelWater.pickle')

TIKI_NAME = 'tiki'
WATER_GAP1_NAME = 'waterGap1'
WATER_GAP2_NAME = 'waterGap2'
TEMPLE_LEVEL_WATER_NAME = 'templeLevelWater'

TEMPLATE = True
FEATURE = False

'''
Each obstacle has a obstacle template for template matching
or SIFT descriptor for feature matching, a name, and a method.
The name and method are to determine how to parse the frame for
the given obstacle.
'''
TEMPLE_OBSTACLES = [(TREE_ROOT1, TREE_ROOT1_NAME, TEMPLATE), 
                    (TREE_ROOT2, TREE_ROOT2_NAME, TEMPLATE),
                    (TREE_ROOT3, TREE_ROOT3_NAME, TEMPLATE),
                    (TREE_ROOT4, TREE_ROOT4_NAME, TEMPLATE),
                    (TREE_TRUNK, TREE_TRUNK_NAME, FEATURE),
                    (GAP1, GAP1_NAME, FEATURE),
                    (GAP2, GAP2_NAME, FEATURE),
                    (FIRE_TRAP, FIRE_TRAP_NAME, FEATURE),
                    (ALTERNATE_LEVEL, ALTERNATE_LEVEL_NAME, FEATURE)]

ALTERNATE_OBSTACLES = [(TIKI, TIKI_NAME, TEMPLATE),
                       (WATER_GAP1, WATER_GAP1_NAME, FEATURE),
                       (WATER_GAP2, WATER_GAP2_NAME, FEATURE),
                       (STONE_GAP1, STONE_GAP1_NAME, FEATURE),
                       (STONE_GAP2, STONE_GAP2_NAME, FEATURE),
                       (STONE_TREE_TRUNK, STONE_TRUNK_NAME, TEMPLATE),
                       (STONE_TREE_TRUNK1, STONE_TRUNK1_NAME, TEMPLATE),
                       (TEMPLE_LEVEL_WATER, TEMPLE_LEVEL_WATER_NAME, FEATURE),
                       (TEMPLE_LEVEL_STONE, TEMPLE_LEVEL_STONE_NAME, FEATURE)]

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

def get_descriptors(img):
    '''
    Description:
        Finds the SIFT descriptors of an image
    Input:
        - img (nd.array): An image
    Output:
        - The SIFT descriptors of the image
    '''
    orb = cv2.SIFT_create()
    _, descriptors = orb.detectAndCompute(img, None)
    return descriptors

def get_descriptor_matches(des1, des2):
    '''
    Description:
        Finds the number of matches between two SIFT descriptors
    Input:
        - des1 (list): List of SIFT descriptors for image one
        - des2 (list): List of SIFT descriptors for image two
    Output:
        - The number of matches between the descriptors
    '''
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1,des2,k=2)
    except:
        return 0

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        except:
            continue
    
    return np.sum(matchesMask)

def jump():
    kb.press('w')
    sleep(0.025)
    kb.release('w')

def slide():
    kb.press('s')
    sleep(0.025)
    kb.release('s')

def check_for_obstacle(frame, debug=0):
    '''
    Loops through the obstacles associated with the current state and makes an action
    if that obstacle is detected.
    '''
    global OBSTACLES
    global TEMPLE_OBSTACLES
    global ALTERNATE_OBSTACLES

    frame_descriptors = get_descriptors(frame)

    for obstacle, name, method in OBSTACLES:
        # algorithm for template matching
        if method == TEMPLATE:
            result = cv2.matchTemplate(frame, obstacle, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

            if (name == TREE_ROOT1_NAME or name == TREE_ROOT4_NAME) and maxVal > 0.7:
                jump()
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            elif name == TREE_ROOT3_NAME and maxVal > 0.75:
                jump()
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            elif name == TREE_ROOT2_NAME and maxVal > 0.70:
                sleep(0.005)  # sleep briefly since it detects this obstacle further back
                jump()
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            # stone trunk not fully working yet
            elif (name == STONE_TRUNK_NAME or name == STONE_TRUNK1_NAME) and maxVal > 0.4:
                slide()
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            elif name == TIKI_NAME and maxVal > 0.40:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

        # algorithm for feature matching
        if method == FEATURE:

            matches = get_descriptor_matches(obstacle, frame_descriptors)

            if name == GAP1_NAME and matches > 60:
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == GAP2_NAME and matches > 60:
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == FIRE_TRAP_NAME and matches > 40:
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == STONE_GAP1_NAME and matches > 50:
                sleep(0.05)
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == STONE_GAP2_NAME and matches > 50:
                sleep(0.05)
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == WATER_GAP1_NAME and matches > 20:
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == WATER_GAP2_NAME and matches > 20:
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == TIKI_NAME and matches > 15:
                jump()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == TEMPLE_LEVEL_WATER_NAME  and matches > 50:
                jump()
                OBSTACLES = TEMPLE_OBSTACLES
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == TEMPLE_LEVEL_STONE_NAME and matches > 20:
                jump()
                OBSTACLES = TEMPLE_OBSTACLES
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == ALTERNATE_LEVEL_NAME and matches > 20:
                sleep(0.25)
                jump()
                OBSTACLES = ALTERNATE_OBSTACLES
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

            elif name == TREE_TRUNK_NAME and matches > 10:
                slide()
                if debug:
                    print(name + ': ' + str(matches))
                    print()
                return

        check_for_turn(frame, OBSTACLES == TEMPLE_OBSTACLES)

        # check_for_tree_trunk(frame, OBSTACLES == TEMPLE_OBSTACLES)


def check_for_turn(frame, temple):
    '''
    Averages the pixels of three patches located on the screen and performs
    a turn based on the results.
    '''
    if temple:
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
    else:
        th, binary = cv2.threshold(frame, 75, 255, cv2.THRESH_BINARY)
        patch0 = binary[150:200, 50:100]
        patch1 = binary[100:150, 250:300]
        patch2 = binary[150:200, 440:490]
        patch0_average = np.average(patch0)
        patch1_average = np.average(patch1)
        patch2_average = np.average(patch2)
        if patch0_average > 55:
            kb.press('a')
            sleep(0.025)
            kb.release('a')
        elif patch2_average > 55:
            kb.press('d')
            sleep(0.025)
            kb.release('d')


def check_for_tree_trunk(frame, temple):
    if temple:
        patch0 = frame[100:150, 100:150]
        patch1 = frame[0:50, 250:300]
        patch2 = frame[100:150, 400:450]
        patch3 = frame[200:250, 250:300]
        patch0_average = np.mean(patch0)
        patch1_average = np.mean(patch1)
        patch2_average = np.mean(patch2)
        patch3_average = np.mean(patch3)
        if patch0_average < 50 and patch1_average < 50 and patch2_average < 50 and patch3_average > 70:
            slide()
            sleep(1)
            # cv2.rectangle(frame, (100, 100), (150, 150), (255, 0, 0), 3)
            # cv2.rectangle(frame, (250, 0), (300, 50), (255, 0, 0), 3)
            # cv2.rectangle(frame, (400, 100), (450, 150), (255, 0, 0), 3)
            # cv2.rectangle(frame, (250, 200), (300, 250), (255, 0, 0), 3)
            # cv2.imshow('Tree Trunk', frame)
            # print(patch0_average, patch1_average, patch2_average, patch3_average)


def on_press(key):
    global OBSTACLES
    global DEBUG
    
    try:
        if (key.char == 'r'):
            OBSTACLES = TEMPLE_OBSTACLES
            print('Reset obstacles...')
        elif (key.char == 'c'):
            OBSTACLES = []
            print('Cleared obstacles...')
        elif (key.char == 'f'):
            DEBUG = not DEBUG
            print('Toggling FPS output...')
    except:
        print('Key not recognized')


def main():
    if RECORDING:
        output = cv2.VideoWriter('ScreenCaptures/temple_run_4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (540, 260), 0)

    loop_time = time()

    listener = Listener(on_press = on_press)
    listener.start()
    
    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # focus on a 260x640 region of the frame
            obstacle_region = grayscale_frame[350:610,:] 
            
            check_for_obstacle(obstacle_region, debug=1)

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