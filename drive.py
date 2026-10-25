import base64                      # the simulator sends the image as base64 text; this decodes it
import os                          # used to build the path to model.h5 and check it exists
from io import BytesIO             # lets us treat raw image bytes like a file in memory

import cv2                         # OpenCV: image processing (crop, colour change, blur, resize)
import eventlet                    # the web server engine the simulator talks to
import eventlet.wsgi               # runs our app on that server
import numpy as np                 # arrays / numbers (images are arrays of pixels)
import socketio                    # Socket.IO: real-time two-way messages with the simulator
from keras.models import load_model  # loads your trained neural network from model.h5
from PIL import Image              # opens the decoded image bytes as a picture

# ---------- Settings you can change ----------
SPEED_LIMIT = 30     # target speed: the car accelerates until it reaches this, then eases off. Higher = faster but riskier (the car may swerve or leave the road).
PORT = 4567          # the simulator always connects to this port - don't change it
# Full path to model.h5, looked up next to this script (so it works from any folder)
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")

# Create the Socket.IO server object. We attach "event handlers" to it below.
sio = socketio.Server()


# ---------- Image preprocessing ----------
def img_preprocess(img):
    """Must be EXACTLY the same steps you used when training in the notebook.
    If these differ even slightly, the model will give bad steering."""
    img = img[60:135, :, :]                      # crop: keep only the road (remove sky on top and car hood at bottom)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)   # convert colours to YUV (what the NVIDIA model expects)
    img = cv2.GaussianBlur(img, (3, 3), 0)       # small blur to reduce noise
    img = cv2.resize(img, (200, 66))             # resize to the model's input size (200 wide, 66 tall)
    return img / 255                             # scale pixel values from 0-255 down to 0-1


# ---------- Event: the simulator sends us data ----------
@sio.on("telemetry")
def telemetry(sid, data):
    """Runs every time the simulator sends a camera picture + speed (~10 times per second)."""
    speed = float(data["speed"])                                    # current car speed (sent as text, so convert to number)
    image = Image.open(BytesIO(base64.b64decode(data["image"])))    # decode the base64 text back into a picture
    image = img_preprocess(np.asarray(image))                       # picture -> numbers -> same preprocessing as training
    image = np.array([image])                                       # model expects a batch of images, so wrap it in a list

    steering_angle = float(model(image, training=False)[0][0])      # ask the model: how much should we steer? (-ve = left, +ve = right)
    throttle = 1.0 - speed / SPEED_LIMIT                            # gas pedal: full gas when slow, easing to 0 as we reach the limit
    print(f"steering={steering_angle:+.3f}  throttle={throttle:+.3f}  speed={speed:.1f}")  # show values in the terminal
    send_control(steering_angle, throttle)                          # send our decision back to the simulator


# ---------- Event: the simulator connects ----------
@sio.on("connect")
def connect(sid, environ):
    """Runs once when the simulator first connects."""
    print("Simulator connected")
    send_control(0, 0)       # start with no steering and no gas so the car doesn't lurch forward


# ---------- Helper: send commands to the simulator ----------
def send_control(steering_angle, throttle):
    """Sends a 'steer' message with the steering angle and throttle to the simulator."""
    sio.emit("steer", data={
        "steering_angle": str(steering_angle),   # the simulator expects text, not numbers
        "throttle": str(throttle),
    })


# ---------- Program starts here ----------
if __name__ == "__main__":
    # Stop early with a clear message if the model file is missing
    if not os.path.exists(MODEL_FILE):
        raise SystemExit(f"Can't find the model file: {MODEL_FILE}\n"
                        "Put model.h5 in the same folder as drive.py.")
    model = load_model(MODEL_FILE, compile=False)   # load the trained network; compile=False because we only predict, not train
    print("Model loaded. Now start the simulator in AUTONOMOUS MODE.")
    app = socketio.WSGIApp(sio)                     # wrap the Socket.IO server as a web app
    eventlet.wsgi.server(eventlet.listen(("", PORT)), app)   # start listening for the simulator on port 4567