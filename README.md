# Real-Time Traffic Sign Recognition (CNN + OpenCV)

> 🇪🇸 Reconocimiento de señales de tránsito en tiempo real con una red neuronal convolucional y la cámara web.

A convolutional neural network built with TensorFlow/Keras that classifies Colombian traffic signs. It runs live on a webcam feed with OpenCV.

## How it works

1. **Training** (`modelo.ipynb`, Google Colab)
   - Images are resized to 160×160 RGB and normalized.
   - A sequential CNN learns to classify 3 sign classes. It has 5 Conv2D + MaxPooling blocks with 25 % dropout, then a Dense(512) layer with 50 % dropout and a softmax output.
   - Training uses the Adam optimizer and categorical cross-entropy for up to 32 epochs, with early stopping and model checkpointing.
   - The trained model is saved as `modelo.h5`.
2. **Detection** (`Detector.py`)
   - OpenCV captures the webcam and draws a region of interest on screen.
   - Press **`s`** to capture the region. The model classifies the sign and the label is drawn over the video.
   - Predictions with less than **85 % confidence** are ignored.
   - An automatic mode (classify every frame) is included in the code, commented out.

```
webcam ──► ROI crop ──► resize 160×160 ──► CNN (Keras) ──► softmax ──► label (if p ≥ 0.85)
```

## Project structure

| Path | Content |
|---|---|
| `modelo.ipynb` | Data loading, model design, training and evaluation |
| `Detector.py` | Real-time detection app |
| `Train/`, `Test/` | Image datasets used for training and testing |

## Run it locally

```bash
pip install tensorflow opencv-python imutils numpy pandas matplotlib scipy
python Detector.py   # needs a webcam and modelo.h5 in the same folder
```

Place the sign inside the blue rectangle and press `s`. Press `Esc` to quit.

## Results and limitations

- The model reached 100 % accuracy on the **training set** and correctly classified a 5-image hand-picked test batch.
- No separate validation split was used, so the training accuracy overstates real-world performance.

## Next steps

- Add a proper train/validation/test split, a confusion matrix and per-class precision and recall.
- Retrain on a public benchmark such as GTSRB to compare results.
- Add a `requirements.txt` and a demo GIF.

## Tech

Python · TensorFlow/Keras · OpenCV · NumPy · scikit-learn (metrics) · Google Colab

---
Author: **Juan Diego Murillo** · [LinkedIn](https://www.linkedin.com/in/juandiegomurillo) · 2023
