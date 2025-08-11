# Emotion Detection
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Vishwajeet3007/Projects/tree/main/Emotion%20Detection)

This project focuses on building and training a deep learning model to detect human emotions from facial expressions. The model is trained to classify faces into seven distinct emotional categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

The repository includes a Jupyter Notebook for training a model using transfer learning with ResNet50 on the FER-2013 dataset, as well as a Python script for real-time emotion detection via a webcam.

## Features
- **Emotion Classification**: Classifies facial expressions into 7 categories.
- **Transfer Learning**: Utilizes a pre-trained ResNet50 model for feature extraction, fine-tuned for emotion detection.
- **Class Imbalance Handling**: Implements class weights during training to address the imbalanced nature of the FER-2013 dataset.
- **Model Evaluation**: Includes code for evaluating the model using metrics like accuracy, recall, and a confusion matrix.
- **Real-Time Detection**: A script (`test.py`) using OpenCV allows for live emotion detection from a webcam feed.

## Dataset
This project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset, which is publicly available on Kaggle. The notebook will automatically download and set up this dataset.

- **Dataset Link**: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset consists of 48x48 pixel grayscale images of faces, which are categorized into seven emotions. For the ResNet50 model in the training notebook, these images are resized to 224x224.

## Project Structure

### Training (`Emotion_Detection.ipynb`)
- **Model**: A transfer learning model based on ResNet50 with a custom classification head.
- **Input Shape**: `(224, 224, 3)`
- **Framework**: TensorFlow / Keras
- **Process**:
    1.  Downloads and prepares the FER-2013 dataset.
    2.  Builds a Keras model by freezing the base ResNet50 layers and adding `GlobalAveragePooling2D`, `Dropout`, and a final `Dense` layer for classification.
    3.  Trains the model for 20 epochs with the Adam optimizer and class weights.
    4.  Saves the best-performing model as `emotion_resnet50.keras`.
    5.  Evaluates the model and provides a confusion matrix for performance analysis.

### Real-Time Inference (`test.py`)
- **Model**: This script is configured to use a separate, pre-trained CNN model named `CNN_Model.keras`.
- **Input Shape**: `(48, 48, 1)` (Grayscale)
- **Framework**: Keras / OpenCV
- **Process**:
    1.  Loads a Haar Cascade classifier for face detection.
    2.  Captures video from the webcam.
    3.  For each frame, detects faces, crops the region of interest (ROI), and preprocesses it (resize to 48x48, convert to grayscale, normalize).
    4.  Feeds the ROI into the loaded model for emotion prediction.
    5.  Overlays the predicted emotion label and confidence score on the video feed.

***Note***: The model trained in `Emotion_Detection.ipynb` (`emotion_resnet50.keras`) has a different architecture and input shape (224x224 RGB) than the one expected by `test.py` (`CNN_Model.keras`, 48x48 grayscale). To use the trained ResNet50 model for real-time detection, you will need to modify `test.py` to handle the different input requirements.

## Getting Started

### Prerequisites
- Python 3.8+
- A Kaggle account and an API token (`kaggle.json`). [How to get Kaggle API token](https://www.kaggle.com/docs/api).
- `haarcascade_frontalface_default.xml` file for `test.py`. You can download it from the [OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vishwajeet3007/Projects.git
    cd Projects/Emotion\ Detection
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### 1. Training the ResNet50 Model
1.  Place your `kaggle.json` file in the root directory (`/content/` if using Google Colab, or the local project directory).
2.  Open and run the `Emotion_Detection.ipynb` notebook in a Jupyter environment or Google Colab.
3.  The notebook will handle downloading the dataset, training the model, and saving the final weights as `emotion_resnet50.keras`.

#### 2. Running Real-Time Emotion Detection
1.  Ensure you have a pre-trained model file named `CNN_Model.keras` in the same directory, or modify the script to load your model (e.g., `emotion_resnet50.keras`) and adjust the preprocessing steps accordingly.
2.  Make sure the `haarcascade_frontalface_default.xml` file is in the same directory.
3.  Run the script from your terminal:
    ```bash
    python test.py
    ```
4.  A window with your webcam feed will appear. The application will detect faces and draw a bounding box with the predicted emotion.
5.  Press `q` to close the application.