from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import google.generativeai as genai
import io
import base64
import os
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
CORS(app)

# 🔹 Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_FALLBACK_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# 🔹 Load the model (use relative path for deployment)
model_path = os.path.join(os.path.dirname(__file__), "model", "heart_disease_mri_densenet_model.h5")
heart_disease_model = tf.keras.models.load_model(model_path)
heart_disease_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 🔹 Classification categories
categories = ["heart_failure_with_infarct", "heart_failure_without_infarct", "hypertrophy", "normal"]

def preprocess_image(image, target_size=(224, 224)):
    try:
        img = Image.open(io.BytesIO(image.read())).convert('RGB')
        img = img.resize(target_size)
        img_array = img_to_array(img) / 255.0
        return img, np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Image processing error: {e}")
        return None, None

def annotate_image(img, predicted_class):
    img = img.convert("RGB")

    text_box_color = "red" if predicted_class in categories[:-1] else "green"
    text_height = 35
    new_width, new_height = img.size[0], img.size[1] + text_height
    new_img = Image.new("RGB", (new_width, new_height), text_box_color)
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    text = f"Predicted: {predicted_class}"
    text_position = (10, img.size[1] + 5)
    draw.text(text_position, text, font=font, fill="white")

    img_io = io.BytesIO()
    new_img.save(img_io, format="PNG")
    img_io.seek(0)

    return base64.b64encode(img_io.read()).decode("utf-8")

@app.route('/')
def home():
    return "🩺 Heart Disease Classifier API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img, img_array = preprocess_image(file)

    if img_array is None:
        return jsonify({'error': 'Error processing the image. Please try again.'}), 400

    predictions = heart_disease_model.predict(img_array)
    predicted_class = categories[np.argmax(predictions, axis=1)[0]]
    annotated_image_base64 = annotate_image(img, predicted_class)

    return jsonify({'prediction': predicted_class, 'image': annotated_image_base64})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get("user_input", "")
    prediction_result = data.get("prediction_result", "")

    if not user_input:
        return jsonify({'response': "Please ask a valid question."}), 400

    prompt = f"""
    You are a medical AI chatbot specializing in heart disease and cardiac MRI analysis.
    The user has received a prediction result: {prediction_result}.
    The user asked: {user_input}

    Provide a detailed, informative, and clear response based on this context.
    """

    try:
        response = model.generate_content([prompt])
        bot_reply = response.text if hasattr(response, "text") else "I'm unable to provide an answer at the moment."
    except Exception as e:
        bot_reply = f"Error: {str(e)}"

    return jsonify({'response': bot_reply})

# 🔹 Don't use debug mode in production
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Default port Render expects
    app.run(host="0.0.0.0", port=port)

