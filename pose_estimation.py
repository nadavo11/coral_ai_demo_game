# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.
To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet
Example usage:
```
bash examples/install_requirements.sh movenet_pose_estimation.py
python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""

import argparse

import numpy as np
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
_NUM_KEYPOINTS = 17
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

RED = (200,20,20)
interpreter = make_interpreter(model)
interpreter.allocate_tensors()
def line(p1, p2):
    pygame.draw.line(screen, WHITE, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))

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




    draw = ImageDraw.Draw(img)
    width, height = img.size
    for i in range(0, _NUM_KEYPOINTS):
        draw.ellipse(
            xy=[
                pose[i][1] * width - 2, pose[i][0] * height - 2,
                pose[i][1] * width + 2, pose[i][0] * height + 2
            ],
            fill=(255, 0, 0))


    #img.save(args.output)
    #img.save(args.output)
    #print('Done. Results saved at', args.output)

    return np.array(img)


import cv2

# define a video capture object
vid = cv2.VideoCapture(1)
def draw_body(pose):


    cv2.line(pose[rightShoulder], pose[leftShoulder],RED,thickness = 4)
    cv2.line(pose[rightShoulder], pose[rightElbow],RED,thickness = 4)
    cv2.line(pose[rightElbow], pose[rightWrist],RED,thickness = 4)
    cv2.line(pose[leftShoulder], pose[leftElbow],RED,thickness = 4)
    cv2.line(pose[leftElbow], pose[leftWrist],RED,thickness = 4)

    cv2.line(pose[rightHip], pose[leftHip],RED,thickness = 4)
    cv2.line(pose[rightShoulder], pose[rightHip],RED,thickness = 4)
    cv2.line(pose[rightKnee], pose[rightHip],RED,thickness = 4)
    cv2.line(pose[leftShoulder], pose[leftHip],RED,thickness = 4)
    cv2.line(pose[leftKnee], pose[leftHip],RED,thickness = 4)
    cv2.line(pose[leftKnee], pose[leftAnkle],RED,thickness = 4)
    cv2.line(pose[rightKnee], pose[rightAnkle],RED,thickness = 4)



while (True):

    # Capture the video frame
    # by frame
    ret, inp = vid.read()
    pose = det_pose(inp)
    draw_body(pose)
    # Display the resulting frame
    cv2.imshow('output',pose)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()