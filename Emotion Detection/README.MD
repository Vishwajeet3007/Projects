🎭 Real-Time Emotion Detection
A real-time facial emotion recognition system using TensorFlow, OpenCV, and Keras.
This project captures live video from your webcam, detects faces, and predicts emotions such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

📌 Features
Real-time face detection using Haar Cascade Classifier

Deep Learning model trained on grayscale 48×48 facial images

Displays predicted emotion with confidence score

Works with live webcam feed

Easy to run locally

📂 Project Structure
bash
Copy
Edit
📁 Emotion-Detection/
│── Emotion_Detection.ipynb    # Jupyter Notebook for training & testing
│── test.py                    # Script to run real-time emotion detection
│── requirements.txt           # Python dependencies
│── haarcascade_frontalface_default.xml  # Face detection model (add this file)
│── CNN_Model.keras            # Pre-trained CNN model
🛠 Requirements
Install the dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt includes:

streamlit

tensorflow

opencv-python

numpy

Pillow

gTTS

playsound==1.2.2

▶️ How to Run
1️⃣ Clone this repository

bash
Copy
Edit
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Make sure you have the model & Haar Cascade file

Place CNN_Model.keras in the root directory

Download Haar Cascade XML from OpenCV GitHub and save as haarcascade_frontalface_default.xml

4️⃣ Run the real-time detection

bash
Copy
Edit
python test.py
5️⃣ Quit the program
Press q to exit the webcam feed.

🧠 Emotion Categories
😠 Angry

🤢 Disgust

😨 Fear

😄 Happy

😐 Neutral

😢 Sad

😲 Surprise

📷 Demo
(Add your screenshot or gif here)

🚀 Future Improvements
Add Streamlit UI for browser-based interaction

Support for multiple faces in a single frame

Improve accuracy with larger datasets and transfer learning