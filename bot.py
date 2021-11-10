import cv2
from time import time, sleep
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mss.windows import MSS as mss
from pynput.keyboard import Controller, Listener

RECORDING = 1          # set this flag high if you want the screen recorded
DEBUG = 0              # set this flag high to enable debugging in the main loop

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

def unpickle_desc(path):
    '''
    Description:
        Unpickles a pickled ORB description object
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

*** TO-DO ***
* Get images of the remaining obstacles
'''
TREE_ROOT1 = cv2.imread('./Obstacles/Temple/treeRoot1.png', 0)
TREE_ROOT2 = cv2.imread('./Obstacles/Temple/treeRoot2.png', 0)
TREE_ROOT3 = cv2.imread('./Obstacles/Temple/treeRoot3.png', 0)
TREE_ROOT4 = cv2.imread('./Obstacles/Temple/treeRoot4.png', 0)
TREE_TRUNK = cv2.imread('./Obstacles/Temple/treeSlide1.png', 0)
GAP1 = unpickle_desc('./Obstacles/Temple/ORB/gap1.pickle')
GAP2 = unpickle_desc('./Obstacles/Temple/ORB/gap2.pickle')
FIRE_TRAP = unpickle_desc('./Obstacles/Temple/ORB/fireTrap.pickle')
ROCK_LEVEL = cv2.imread('./Obstacles/Temple/rockLevel.png', 0)
ROCK_LEVEL2 = cv2.imread('./Obstacles/Temple/rockLevel2.png', 0)
ALTERNATE_LEVEL = cv2.imread('./Obstacles/Temple/waterLevel.png', 0)

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

*** TO-DO ***
* Get images of the remaining obstacles
'''
STONE_TREE_TRUNK = cv2.imread('./Obstacles/Water/stoneTrunk.png', 0)
STONE_GAP = cv2.imread('./Obstacles/Water/stoneGap.png', 0)

STONE_TRUNK_NAME = 'treeTrunk'

'''
Water Obstacle Templates

*** TO-DO ***
* Get images of the remaining obstacles
'''
TIKI = cv2.imread('./Obstacles/Water/stoneGate.png', 0)
WATER_GAP = cv2.imread('./Obstacles/Water/waterGap.png', 0)
TEMPLE_LEVEL = cv2.imread('./Obstacles/Water/templeLevel.png', 0)

TIKI_NAME = 'tiki'
TEMPLE_LEVEL_NAME = 'templeLevel'

TEMPLATE = True
FEATURE = False
'''
Each obstacle has a obstacle template for template matching
or ORB descriptor for feature matching, a name, and a method.
The name and method are to determine how to parse the frame for
the given obstacle.
'''
TEMPLE_OBSTACLES = [(TREE_ROOT1, TREE_ROOT1_NAME, TEMPLATE), 
                    (TREE_ROOT2, TREE_ROOT2_NAME, TEMPLATE),
                    (TREE_ROOT3, TREE_ROOT3_NAME, TEMPLATE),
                    (TREE_ROOT4, TREE_ROOT4_NAME, TEMPLATE),
                    (TREE_TRUNK, TREE_TRUNK_NAME, TEMPLATE), 
                    (GAP1, GAP1_NAME, FEATURE),
                    (GAP2, GAP2_NAME, FEATURE),
                    (FIRE_TRAP, FIRE_TRAP_NAME, FEATURE),
                    (ALTERNATE_LEVEL, ALTERNATE_LEVEL_NAME, TEMPLATE)]

ALTERNATE_OBSTACLES = [(TIKI, TIKI_NAME, TEMPLATE),
                       (WATER_GAP, 'waterGap', 'template'),
                       (STONE_TREE_TRUNK, STONE_TRUNK_NAME, TEMPLATE),
                       (TEMPLE_LEVEL, TEMPLE_LEVEL_NAME, TEMPLATE)]

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
        Finds the ORB descriptors of an image
    Input:
        - img (nd.array): An image
    Output:
        - The ORB descriptors of the image
    '''
    orb = cv2.ORB_create()
    _, descriptors = orb.detectAndCompute(img, None)
    return descriptors

def get_descriptor_matches(des1, des2):
    '''
    Description:
        Finds the matches between two ORB descriptors
    Input:
        - des1 (list): List of ORB descriptors for image one
        - des2 (list): List of ORB descriptors for image two
    Output:
        - The number of matches between the descriptors
    '''
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 2) #2
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

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

def check_for_obstacle(frame, debug=0):
    '''
    Loops through the obstacles associated with the current state and makes an action
    if that obstacle is detected.
    '''
    global OBSTACLES
    global TEMPLE_OBSTACLES
    global ALTERNATE_OBSTACLES

    for obstacle, name, method in OBSTACLES:
        # algorithm for template matching
        if method == TEMPLATE:
            result = cv2.matchTemplate(frame, obstacle, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

            if (name == FIRE_TRAP_NAME or name == TREE_ROOT1_NAME or name == TREE_ROOT4_NAME) and maxVal > 0.7:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            elif name == TREE_ROOT3_NAME and maxVal > 0.75:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            elif name == TREE_ROOT2_NAME and maxVal > 0.60:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            # stone trunk not fully working yet
            elif (name == TREE_TRUNK_NAME or name == STONE_TRUNK_NAME) and maxVal > 0.40:
                kb.press('s')
                sleep(0.025)
                kb.release('s')
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            elif name == TIKI_NAME and maxVal > 0.4:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                    print()
                return

            # alternate level jump not always working for stone
            elif name == ALTERNATE_LEVEL_NAME and maxVal > 0.45:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                OBSTACLES = ALTERNATE_OBSTACLES
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                return

            # works *most* of the time, might want to use ORB for this
            elif name == TEMPLE_LEVEL_NAME and maxVal > 0.5:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                OBSTACLES = TEMPLE_OBSTACLES
                if debug:
                    display_template_match(frame, obstacle, maxLoc)
                    print(maxVal, name)
                return

        # algorithm for feature matching
        if method == FEATURE:

            feature_frame = np.copy(frame)

            # crop frame based on current obstacle (crop dimensions 
            # obtained through experimental measurments)
            if name == GAP1_NAME or name == GAP2_NAME:
                feature_frame = feature_frame[75:175, 150:350]

            elif name == FIRE_TRAP_NAME:
                feature_frame = feature_frame[20:175, 100:200]
            
            frame_descriptors = get_descriptors(feature_frame)
            matches = get_descriptor_matches(obstacle, frame_descriptors)

            if (name == GAP1_NAME or name == GAP2_NAME) and matches > 20:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                if debug:
                    print(name + ': ' + str(matches))
                return

            elif name == FIRE_TRAP_NAME and matches > 20:
                kb.press('w')
                sleep(0.025)
                kb.release('w')
                if debug:
                    print(name + ': ' + str(matches))
                return

        check_for_turn(frame, OBSTACLES == TEMPLE_OBSTACLES)


def check_for_turn(frame, temple):
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
    if temple:
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
        # still needs tweaking, might need to separate stone and water for this part
        if patch0_average > 50:
            kb.press('a')
            sleep(0.025)
            kb.release('a')
        elif patch2_average > 50:
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