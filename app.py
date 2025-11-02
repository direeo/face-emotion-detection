# app.py
import os
import sqlite3
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
# TensorFlow is optional for running the web UI; import lazily and fail gracefully
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except Exception:
    load_model = None
    image = None
    TF_AVAILABLE = False
import numpy as np
from PIL import Image

# Config
UPLOAD_FOLDER = 'uploads'
DB_PATH = 'database.db'
MODEL_PATH = 'face_emotionModel.h5'
LABEL_MAP_PATH = 'label_map.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = 48

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # for flash messages (change in production)

# Load model and labels
model = None
if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print("Warning: failed to load model:", e)
        model = None
else:
    if not TF_AVAILABLE:
        print("TensorFlow not available: running without model. Install TensorFlow to enable predictions.")
    else:
        print("Warning: model file not found at", MODEL_PATH)

if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, 'r') as f:
        LABELS = json.load(f)
else:
    LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map an emotion label to a friendly sentence
FRIENDLY_MESSAGES = {
    'Angry': "You look angry. Take a breath — is everything okay?",
    'Disgust': "You look disgusted. Want to tell me what's bothering you?",
    'Fear': "You look afraid. You're safe here — what's worrying you?",
    'Happy': "You're smiling! That's great — what's making you happy?",
    'Sad': "You are frowning. Why are you sad?",
    'Surprise': "You look surprised! What happened?",
    'Neutral': "You look calm and neutral."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_prediction(img_path):
    img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype='float32') / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    arr = np.expand_dims(arr, axis=0)   # (1,H,W,1)
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'photo' not in request.files:
        return "No file uploaded", 400

    file = request.files['photo']
    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return "File type not allowed. Use: .png, .jpg, .jpeg, or .gif", 400

    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Get form data
    name = request.form.get('name', '').strip()
    student_id = request.form.get('student_id', '').strip()
    email = request.form.get('email', '').strip()

    # Do prediction
    if model is None:
        emotion = "Model not found"
    else:
        try:
            x = preprocess_image_for_prediction(filepath)
            preds = model.predict(x)
            idx = int(np.argmax(preds))
            emotion = LABELS[idx]
        except Exception as e:
            print("Prediction error:", e)
            emotion = "Error during prediction"

    # Get friendly message
    message = FRIENDLY_MESSAGES.get(emotion, f"Predicted: {emotion}")

    # Store in database
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS submissions
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT, student_id TEXT, email TEXT,
                     filename TEXT, predicted_emotion TEXT,
                     message TEXT)''')
        c.execute('INSERT INTO submissions (name, student_id, email, filename, predicted_emotion, message) VALUES (?, ?, ?, ?, ?, ?)',
                 (name, student_id, email, filename, emotion, message))
        conn.commit()
    except Exception as e:
        print("Database error:", e)
    finally:
        if 'conn' in locals():
            conn.close()

    # Render the index page with the prediction so result shows on the same page
    return render_template('index.html', message=message, emotion=emotion, filename=filename)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("\nStarting server...")
    print("Open http://127.0.0.1:5000 in your web browser")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5000)
