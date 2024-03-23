# flask
from flask import Flask, request, jsonify
import tensorflow as tf

# converting image
from PIL import Image
import io
import base64
import re
import numpy as np

# keras
from keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)


# load trained model
model_path = ".venv/sdgp/model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define category labels
category_labels = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections",
]

prediction_url = "http://localhost:3000/predict"
print(f"Prediction requests can be sent to: {prediction_url}")

# define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.json:
        print("No image data found in request")
        return jsonify("No image data found")

    # Extract base64 image data
    base64_img = request.json["image"]
    base64_data = re.sub("^data:image/.+;base64,", '', base64_img)

    # Decode base64 image data
    try:
        image_data = base64.b64decode(base64_data)
    except Exception as e:
        print(f"Error decoding base64 data: {e}")
        return jsonify({"error": "Error decoding base64 data"})

    # Convert binary data to PIL image
    image = Image.open(io.BytesIO(image_data))
    # Preprocess the image (adjust based on your model's requirements)
    image = image.resize((192,192))  # Resize image
    img_array = np.asarray(image)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_array = img_array / 255.0

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Invoke the interpreter
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_category_index = np.argmax(output_data[0])
    predicted_category_label = category_labels[predicted_category_index]

    # Return the processed result as a response
    return jsonify({"result": predicted_category_label})


# Add a new route to say hello
@app.route("/")
def hello_world():
    return jsonify({"message": "Hello! This is a skin disease prediction app."})


if __name__ == "__main__":
    app.run(port=3000, debug=True)
