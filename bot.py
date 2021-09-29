import cv2
from mss import mss
from PIL import Image
import numpy as np

recording = 1   # set this flag high if you want the screen recorded
dim = {'top': 0, 'left': 0, 'width': 540, 'height': 960}

def main():
    # can vary the third parameter (FPS) to change the speed of the video
    output = cv2.VideoWriter('ScreenCaptures/temple_run_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (540, 960))

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(dim))[:,:,:3]

            if recording:
                output.write(frame)
            else:
                cv2.imshow('test', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                output.release()
                break

if __name__ == "__main__":
    main()