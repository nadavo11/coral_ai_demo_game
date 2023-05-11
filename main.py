import pygame
import random
import numpy as np
import os
import cv2
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
from pygame.locals import *

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
class Baloon():
    def __init__(self):

        self.radius = 40
        self.x = np.array([width / 2,height / 2])
        self.v = np.array([0.1,2])
        self.score = 0

    def bounce(self,loc):
        dist = np.linalg.norm(self.x - loc[:2])
        if dist-np.linalg.norm(self.v) < self.radius:
            self.v +=  0.3 * (self.x - loc[:2])
            self.score += 1
        return 0

    def show(self):
        # Draw the balloon
        pygame.draw.circle(screen, (150 - self.score,60,self.score), (int(baloon.x[0]),int(baloon.x[1])), baloon.radius)

    def wall_bounce(self):
        # Check if balloon hits the ground
        if self.x[1] + self.radius >= height:
            self.x[1] -= 6
            self.v *= [1, -1]
        if self.x[1] < 0:
            self.v *= [1, -0.5]
            self.x[1] += 4

        if self.radius >= self.x[0] or self.x[0] >= width - self.radius:
            self.v *= [-1, 1]


    def update(self):

        self.bounce(pose[rightShoulder])
        self.bounce(pose[leftShoulder])

        self.bounce(pose[rightWrist])
        self.bounce(pose[leftWrist])

        self.bounce(pose[rightElbow])
        self.bounce(pose[leftElbow])

        self.bounce(pose[rightKnee])
        self.bounce(pose[leftKnee])

        self.bounce(pose[rightAnkle])
        self.bounce(pose[leftAnkle])
        self.bounce(pose[nose])


        self.wall_bounce()
        self.v += g
        self.v *= 0.99
        # Move the balloon
        self.x += self.v

def line(p1,p2):
    pygame.draw.line(screen, WHITE, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])))



def draw_body(pose):
    for p in pose:
        pygame.draw.circle(screen, RED, (int(p[0]), int(p[1])), 5)

    line(pose[rightShoulder],pose[leftShoulder])
    line(pose[rightShoulder],pose[rightElbow])
    line(pose[rightElbow],pose[rightWrist])
    line(pose[leftShoulder],pose[leftElbow])
    line(pose[leftElbow],pose[leftWrist])

    line(pose[rightHip], pose[leftHip])
    line(pose[rightShoulder], pose[rightHip])
    line(pose[rightKnee], pose[rightHip])
    line(pose[leftShoulder], pose[leftHip])
    line(pose[leftKnee], pose[leftHip])
    line(pose[leftKnee],pose[leftAnkle])
    line(pose[rightKnee], pose[rightAnkle])
    
    
    pygame.draw.circle(screen, WHITE, (int(pose[0][0]), int(pose[0][1])), 30)

def get_pose(frame):
    #POSE DETECTION
    pose = det_pose(frame)
    pose[:,1] ,pose[:,0] = pose[:,0]* height ,(1-pose[:,1]) *width
    return pose

def update():
    screen.fill(BLACK)
    draw_body(pose)
    # Bounce balloon if it hits the mouse
    baloon.update()
    baloon.show()
    # Update the display




# define a video capture object
vid = cv2.VideoCapture(1)
# Game loop
running = True
baloon = Baloon()
g = (0,0.5)

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    pose = get_pose(frame)
    update()
    baloon.show()

    pygame.display.flip()

# Quit the game
pygame.quit()
