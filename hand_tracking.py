import os
import cv2
import numpy as np
#from PIL import Image
#from pygame.locals import *
from threading import Thread
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import mediapipe as mp
from PIL import Image
import time
width, height = 1400, 600
"""______________________________________________________________________________
    *                                                                           *
    *                                                                           *
    *                          camStream setup                                  *
    *                                                                           *
    *___________________________________________________________________________*
"""
OPENCV_LOG_LEVEL=0
WIDTH, HEIGHT = 1080, 1920
GREEN = (0, 255, 0)
RED = (0, 0, 255)

class WebcamStream:
    # initialization method
    def __init__(self, stream_id=1):
        self.stream_id = stream_id  # default is 1 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        self.frame_ready = False
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())

        self.t.daemon = True  # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            self.frame_ready = True

            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break

        self.vcap.release()




    # method to return latest read frame
    def read(self):
        return self.frame


    # method to stop reading frames
    def stop(self):
        self.stopped = True




#drawing = mp.solutions.drawing_utils

# initializing and starting multi-threaded webcam input stream
webcam_stream = WebcamStream(stream_id=1) # 0 id for main camera
webcam_stream.start()
# processing frames in input stream
num_frames_processed = 0

# define a video capture object
# vid = cv2.VideoCapture(0)
p = np.zeros((21,3))
""""
    ___________________________________________________________________________
    *                                                                          *
    *                        save video setup                                  *
    *                                                                          *
    *__________________________________________________________________________*
"""
save = False

if save == True:
    # Specify the output file name, codec, frames per second (FPS), and frame size
    output_file = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'x264')  # For MP4 format, use 'mp4v' or 'x264' codec
    fps = 30.0  # Adjust the frames per second according to your requirement
    frame_width = WIDTH  # Get the frame width from the capture object
    frame_height = HEIGHT # Get the frame height from the capture object

    # Create the VideoWriter object
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

"""______________________________________________________________________________
    *                                                                           *
    *                                                                           *
    *                               main Loop                                   *
    *                                                                           *
    *___________________________________________________________________________*
"""
model = "hand_landmark_full.tflite"
interpreter = make_interpreter(model)
interpreter.allocate_tensors()
def det_pose(input):
    """
    takes an image as input and returns a tensor of detected bodypoints in the image.
    A pose is a set of keypoints that represent the position and orientation of a person or an object.
    Each keypoint is a tuple of (x, y), *relative* coordinates of the keypoint,
    The function uses a pre-trained model to perform pose estimation on the image.
    :param input: img
    :return:
    """
    time1 = time.time()
    img = Image.fromarray(input)
    resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
    common.set_input(interpreter, resized_img)

    interpreter.invoke()
    pose = common.output_tensor(interpreter, 0).copy().reshape(21, 3)
    print("detection time:", time.time()-time1)
    return pose


def get_pose(frame):
    # POSE DETECTION
    pose = det_pose(frame)
    pose[:, 1], pose[:, 0] = pose[:, 0] * height, (1 - pose[:, 1]) * width
    return pose


start_time = time.time()
while (True):

    # Capture the video frame
    # by frame
    frame = webcam_stream.read()

    pose = get_pose(frame)
    print(pose)

    run_time  =  time.time()
    print(run_time-start_time)
    start_time = run_time

    # Display the resulting frame
    #cv2.imshow('output', frame)
    #if save:
    #    out.write(frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
webcam_stream.vcap.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()