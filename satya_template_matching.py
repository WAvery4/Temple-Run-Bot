import cv2
from mss import mss
from PIL import Image
import numpy as np

dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

def main():
    template = cv2.imread('fire_template.jpg')

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))[:,:,:3]

            cv2.imshow('test', frame)

            imageGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
            frame_copy = frame.copy()

            if maxVal > 0.4:
                # determine the starting and ending (x, y)-coordinates of the bounding box
                (startX, startY) = maxLoc
                endX = startX + template.shape[1]
                endY = startY + template.shape[0]

                # draw the bounding box on the image
                cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (255, 0, 0), 3)
            cv2.imshow('detect', frame_copy)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()