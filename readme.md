# Self-Driving Car: End-to-End Behavioural Cloning

A convolutional neural network that learns to steer a car **directly from camera images**. It is trained on human driving data from the Udacity Self-Driving Car Simulator, then drives the simulated car on its own in real time.

The model follows the architecture from NVIDIA's paper [*End to End Learning for Self-Driving Cars*](https://arxiv.org/abs/1604.07316). The input is one front-camera frame. The output is one number, the steering angle.

---

## 🎥 Demo

The car driving autonomously on the simulator track (click to watch on YouTube):

[![Self-Driving Car Demo](https://img.youtube.com/vi/Hrakk4kCmLY/hqdefault.jpg)](https://youtu.be/Hrakk4kCmLY)

---

## Features

- **End-to-end learning**: raw pixels in, steering angle out, with no hand-written lane detection.
- **Dataset balancing**: undersamples the over-represented "drive straight" examples so the model doesn't just learn to go straight.
- **Multi-camera training**: uses the left and right cameras with a ±0.15 steering correction to teach the car how to recover when it drifts.
- **On-the-fly augmentation**: random pan, zoom, brightness change and horizontal flip (with the steering angle inverted).
- **Memory-friendly training**: a Python batch generator streams images instead of loading everything into RAM.
- **Real-time inference server**: `drive.py` talks to the simulator over Socket.IO and sends back steering and throttle.

---

## How It Works

### 1. Data

The training data is a recorded driving session (`driving_log.csv` + `IMG/` folder) from the Udacity simulator. It is cloned from [rslim087a/track](https://github.com/rslim087a/track). The log holds the center, left and right camera images with the steering, throttle, reverse and speed values.

| Stage                                               | Samples     |
| --------------------------------------------------- | ----------- |
| Raw driving log                                     | 4,053       |
| After balancing (max 400 per steering bin, 25 bins) | 1,463       |
| After adding left and right cameras                 | 4,389       |
| Training / Validation split (80/20)                 | 3,511 / 878 |

### 2. Augmentation (training only)

Each of the four augmentations is applied independently with 50% probability:

| Augmentation                   | Purpose                         |
| ------------------------------ | ------------------------------- |
| Pan (±10% shift)              | Simulates off-center viewpoints |
| Zoom (1.0 – 1.3×)            | Different apparent distances    |
| Brightness (0.2 – 1.2×)      | Shadows and sunlight            |
| Horizontal flip + invert angle | Removes left/right turning bias |

### 3. Preprocessing

Applied identically in training **and** in `drive.py`:

1. Crop the image to rows 60–135 (removes sky and car hood)
2. Convert RGB → YUV
3. Gaussian blur (3×3)
4. Resize to **200 × 66**
5. Normalize to `[0, 1]`

### 4. Model architecture (NVIDIA)

| Layer                | Details                                        |
| -------------------- | ---------------------------------------------- |
| Input                | 66 × 200 × 3                                 |
| Conv2D ×3           | 24, 36, 48 filters, 5×5 kernel, stride 2, ELU |
| Conv2D ×2           | 64 filters, 3×3 kernel, ELU                   |
| Flatten              |                                                |
| Dense + Dropout(0.5) | 100 units, ELU                                 |
| Dense + Dropout(0.5) | 50 units, ELU                                  |
| Dense + Dropout(0.5) | 10 units, ELU                                  |
| Output               | 1 unit (steering angle)                        |

- **Loss:** Mean Squared Error
- **Optimizer:** Adam, learning rate `1e-4`
- **Training:** 10 epochs, batch size 100, 300 steps per epoch, 200 validation steps

### 5. Results

| Epoch | Train loss | Validation loss |
| ----- | ---------- | --------------- |
| 1     | 0.1177     | 0.0808          |
| 5     | 0.0813     | 0.0623          |
| 10    | 0.0762     | 0.0615          |

Training loss is higher than validation loss because dropout and augmentation are active during training only.

---

## Getting Started

### Prerequisites

- Python 3.9 – 3.12
- The [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) (Term 1 version)

### Installation

```bash
git clone https://github.com/ayushh9999/self-driving-car.git
cd self-driving-car

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> ⚠️ **Do not upgrade `python-socketio` or `python-engineio`.** The Udacity simulator only speaks the old Socket.IO protocol, so `requirements.txt` pins `python-socketio==4.6.1` and `python-engineio==3.13.2`.

### Run the self-driving car

1. Start the driving server:

   ```bash
   python drive.py
   ```

   You should see `Model loaded. Now start the simulator in AUTONOMOUS MODE.`
2. Open the Udacity simulator, choose a track, and click **AUTONOMOUS MODE**.
3. The car starts driving, and the terminal prints the steering angle, throttle and speed live.

### Settings

At the top of `drive.py`:

| Variable        | Default  | Description                                      |
| --------------- | -------- | ------------------------------------------------ |
| `SPEED_LIMIT` | `30`   | Target speed. Higher is faster but riskier.      |
| `PORT`        | `4567` | Port the simulator connects to. Don't change it. |

Throttle is computed as `1 - speed / SPEED_LIMIT`, so the car accelerates when slow and eases off as it approaches the limit.

---

## Train Your Own Model

1. Open `self_driving_car.ipynb` in Google Colab or Jupyter.
2. Run all cells. The notebook clones the dataset, balances and augments it, and trains the network.
3. The model is saved as `model.h5` (and `char_model.keras`). Put `model.h5` next to `drive.py`.

To use your own data, record a lap in the simulator's **Training Mode** and point the notebook's `datadir` at your `driving_log.csv` and `IMG/` folder.

> The preprocessing in `drive.py` **must exactly match** the preprocessing used during training. Even a small difference will degrade the steering.

---

## Tech Stack

Python · TensorFlow / Keras · OpenCV · NumPy · Pandas · imgaug · Scikit-learn · Socket.IO · Eventlet

## Acknowledgements

- [NVIDIA: End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)
- Training dataset from [rslim087a/track](https://github.com/rslim087a/track)
