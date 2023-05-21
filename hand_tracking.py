
import os
import cv2
import numpy as np
#from PIL import Image
#from pygame.locals import *
from threading import Thread

import mediapipe as mp

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


hands = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils

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
while (True):

    # Capture the video frame
    # by frame
    frame = cv2.flip(webcam_stream.read(),1)

    pose = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    if pose.multi_hand_landmarks:
        for hand_landmark in pose.multi_hand_landmarks:
            #drawing.draw_landmarks(frame,hand_landmark,connections=mp.solutions.hands.HAND_CONNECTIONS)
            for i, mark in enumerate(hand_landmark.landmark):
                p[i,0], p[i,1],p[i,2] = mark.x * HEIGHT, mark.y * WIDTH , mark.z * HEIGHT

            if np.linalg.norm(p[10]-p[6]) > 1.3 * np.linalg.norm(p[9]-p[5]):
                txt = "warning"
                color = RED
            else:
                txt = "ok"
                color = GREEN

            cv2.putText(frame, txt, (int(p[10,0]),int(p[10,1])), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2, cv2.LINE_AA)

            #
            # for i, mark in enumerate (hand_landmark.landmark):
            #
            #     if i == 10:
            #         print(frame.shape)
            #         txt = "ok"
            #         color = GREEN
            #
            #         cx, cy = int(mark.x * WIDTH), int(mark.y * HEIGHT)
            #         cv2.putText(frame, txt,(cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('output', frame)
    if save:
        out.write(frame)

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
