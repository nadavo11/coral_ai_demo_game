import pygame
import random
import numpy as np
import os
import cv2
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image

os.environ["DISPLAY"] = ":0"

# Initialize Pygame
#pygame.init()

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

_NUM_KEYPOINTS = 17
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
        self.v = np.array([0.1,0])

    def bounce(self,loc):
        dist = np.linalg.norm(self.x - loc)
        if dist < self.radius:
            self.v +=  0.9 * (self.x - loc)
        return 0

    def show(self):
        # Draw the balloon
        pygame.draw.circle(screen, RED, (int(baloon.x[0]),int(baloon.x[1])), baloon.radius)

    def update(self):
        # Check if balloon hits the ground
        if self.x[1] + self.radius >= height:
            self.v *= [1, -1]
        if self.x[1] < 0:
            self.v *= [1, -0.5]
            self.x[1] += 2

        if self.radius >= self.x[0] or self.x[0] >= width - self.radius:
            self.v *= [-1, 1]

        self.v += g
        self.v *= 0.999
        # Move the balloon
        self.x += self.v

def line(p1,p2):
    pygame.draw.line(screen, WHITE, p1[:-1], p2[:-1])


# define a video capture object
vid = cv2.VideoCapture(1)
# Game loop
running = True
baloon = Baloon()
g = (0,0.5)

while running:
    screen.fill(BLACK)

    # Capture the video frame
    # by frame
    ret, inp = vid.read()
    #POSE DETECTION
    pose = det_pose(inp)
    pose[:,1] ,pose[:,0] = pose[:,0]* height ,pose[:,1] *width
    for p in pose:
    
        pygame.draw.circle(screen, RED, (int(p[0]), int(p[1])), 5)


    #line(pose[rightShoulder],pose[leftShoulder])

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get mouse position
    mouse = pygame.mouse.get_pos()

    # Bounce balloon if it hits the mouse
    baloon.bounce((int(pose[10][0]),int(pose[10][1])))
    baloon.bounce(pose[9])
    baloon.update()
    baloon.show()

    # Update the display
    pygame.display.flip()


# Quit the game
pygame.quit()
