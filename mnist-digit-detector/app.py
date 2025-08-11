import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="MNIST Digit Detector", page_icon="🔢")

@st.cache_resource(show_spinner=False)
def load_model(path="mnist_model.h5"):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img: Image.Image):
    img = img.convert("L")  # Grayscale
    img = ImageOps.fit(img, (28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)

    # Auto invert if background is white
    if arr.mean() > 127:
        arr = 255 - arr

    arr = arr / 255.0  # Normalize
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def predict(model, arr):
    probs = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1]
    return probs, top_idx

def plot_probabilities(probs):
    fig, ax = plt.subplots()
    ax.bar(range(10), probs, color='skyblue')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

def main():
    st.title("🔢 MNIST Digit Detector (Keras Model)")
    st.write("Upload an image of a handwritten digit (0–9) and the model will predict the digit.")

    model = load_model("mnist_model.h5")
    if model is None:
        return

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

        # Auto predict after upload
        with st.spinner("Predicting..."):
            arr = preprocess_image(image)
            probs, top_idx = predict(model, arr)
            pred = int(top_idx[0])

        st.success(f"✅ Predicted Digit: **{pred}**")
        
        # Show probability chart
        plot_probabilities(probs)

        # Show preprocessed image
        st.markdown("### Preprocessed 28×28 Image")
        pre_img = (arr[0].reshape(28, 28) * 255).astype(np.uint8)
        st.image(pre_img, width=140)

    else:
        st.info("Please upload a digit image (0–9) to get predictions.")

if __name__ == "__main__":
    main()
