import pygame
import random
import numpy as np
import os
import cv2
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
from pygame.locals import *
from threading import Thread


_SCORE = 0
os.environ["DISPLAY"] = ":0"
flags = FULLSCREEN | DOUBLEBUF

# Set up the display
width, height = 1400, 600


screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Balloon Bounce")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (180, 40, 40)
GREEN = (40,160,40)

nose = 0
leftEye = 1
rightEye = 2
leftEar = 3
rightEar = 4
leftShoulder = 5
rightShoulder = 6
leftElbow = 7
rightElbow = 8
leftWrist = 9
rightWrist = 10
leftHip = 11
rightHip = 12
leftKnee = 13
rightKnee = 14
leftAnkle = 15
rightAnkle = 16

_NUM_KEYPOINTS = 17
model = "movenet.tflite"
interpreter = make_interpreter(model)
interpreter.allocate_tensors()

_DELTA = 50
_P = 1/150
def det_pose(input):
    """
    takes an image as input and returns a tensor of detected bodypoints in the image.
    A pose is a set of keypoints that represent the position and orientation of a person or an object.
    Each keypoint is a tuple of (x, y), *relative* coordinates of the keypoint,
    The function uses a pre-trained model to perform pose estimation on the image.
    :param input: img
    :return:
    """

    img = Image.fromarray(input)
    resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
    common.set_input(interpreter, resized_img)

    interpreter.invoke()
    pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
    return pose


# Set up the balloon
class Fruit():
    def __init__(self):

        self.radius = 40
        self.x = np.array([width / 2, height / 2])
        self.v = np.array([0.1, 2])
        self.mask = circleMask((255, 255, 255), self.radius) # (color), radius
        self.color = RED

    def show(self):
        # Draw the balloon
        pygame.draw.circle(screen, self.color,
                           (int(self.x[0]), int(self.x[1])), self.radius)

    def cut(self):
        self.color = GREEN
        # TODO remove from ram

    def update(self):

        self.v += g
        self.v *= 0.99
        # Move the balloon
        self.x += self.v


def line(p1, p2):
    pygame.draw.line(screen, WHITE, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))

def circleMask(color, radius):
    shape_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, color, (radius, radius), radius)
    return pygame.mask.from_surface(shape_surf)

class Player:
    def __init__(self):
        #previous hand location
        self.hands_prev = [[0,0],[0,0]]
        #  hand location
        self.hands = [[0,0],[0,0]]
        self.mask =  circleMask((255, 255, 255), 5)

    def cut(self,i):
        line(self.hands_prev[i],self.hands[i])
        for fruit in fruits:
            if fruit.mask.overlap(self.mask, self.hands[i] - fruit.x):
                fruit.cut()


    def update(self):
        # save previous hand location
        self.hands_prev = self.hands
        self.hands = [pose[rightWrist][:2], pose[leftWrist][:2]]

        for i in range(2):
            if np.linalg.norm(self.hands[i] - self.hands_prev[i]) > _DELTA:
                self.cut(i)

def fruit_generator():
    if np.random.rand()< _P:
        fruits. append(Fruit())

def draw_body(pose):
    for p in pose:
        pygame.draw.circle(screen, RED, (int(p[0]), int(p[1])), 5)

    line(pose[rightShoulder], pose[leftShoulder])
    line(pose[rightShoulder], pose[rightElbow])
    line(pose[rightElbow], pose[rightWrist])
    line(pose[leftShoulder], pose[leftElbow])
    line(pose[leftElbow], pose[leftWrist])

    line(pose[rightHip], pose[leftHip])
    line(pose[rightShoulder], pose[rightHip])
    line(pose[rightKnee], pose[rightHip])
    line(pose[leftShoulder], pose[leftHip])
    line(pose[leftKnee], pose[leftHip])
    line(pose[leftKnee], pose[leftAnkle])
    line(pose[rightKnee], pose[rightAnkle])

    pygame.draw.circle(screen, WHITE, (int(pose[0][0]), int(pose[0][1])), 30)


def get_pose(frame):
    # POSE DETECTION
    pose = det_pose(frame)
    pose[:, 1], pose[:, 0] = pose[:, 0] * height, (1 - pose[:, 1]) * width
    return pose
def draw_hands():
    pygame.draw.circle(screen, RED, (int(pose[leftWrist][0]), int(pose[leftWrist][1])), 5)
    pygame.draw.circle(screen, RED, (int(pose[rightWrist][0]), int(pose[rightWrist][1])), 5)

def update():
    screen.fill(BLACK)
    # draw_body(pose)

    fruit_generator()
    for fruit in fruits:
        fruit.update()
        fruit.show()

    player.update()
    draw_hands()

    # Update the display


"""______________________________________________________________________________
    *                                                                           *
    *                                                                           *    
    *                          camStream setup                                  *    
    *                                                                           *       
    *___________________________________________________________________________*   
"""


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



# initializing and starting multi-threaded webcam input stream
webcam_stream = WebcamStream(stream_id=1) # 0 id for main camera
webcam_stream.start()
# processing frames in input stream


num_frames_processed = 0
running = True
fruits = []
player = Player()
g = (0, 0.5)
"""______________________________________________________________________________
    *                                                                           *
    *                                                                           *    
    *                               Game loop                                   *    
    *                                                                           *       
    *___________________________________________________________________________*   
"""

while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read()
    """ --------------
    game handling
    --------------"""
    pose = get_pose(frame)
    update()
    pygame.display.flip()

    # displaying frame
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


webcam_stream.stop() # stop the webcam stream