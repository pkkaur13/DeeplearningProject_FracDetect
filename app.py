from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
model = load_model("model.keras")

print("Model exists:", os.path.exists("model.keras"))
import io
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def prepare_image(file):
    try:
        img = Image.open(io.BytesIO(file.read()))  
        img = img.resize((224, 224))  
        img_array = img_to_array(img)  
        img_array = np.expand_dims(img_array, axis=0) / 255.0  
        return img_array
    except Exception as e:
        print(f"Error preparing image: {str(e)}")
        raise e


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No file selected"}), 400

    try:
        # Check file received
        print(f"Received file: {file.filename}")

        # Proceed with processing the image
        img = prepare_image(file)
        print("Image prepared successfully.")

        # Predict
        prediction = model.predict(img)[0][0]
        label = "Fracture" if prediction > 0.5 else "No Fracture"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        print(f"Prediction: {label}, Confidence: {confidence}")
        return jsonify({"label": label, "confidence": float(confidence)})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
