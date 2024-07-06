from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('soil_model.h5')

class_names = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/model')
def home():
    return render_template('model.html')

@app.route('/about')
def about():
    return render_template('about.html')

def soil_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    return img

@app.route('/predict', methods=['POST'])
def classify_image():
    print("Request received")
    if 'image' not in request.files:
        print("No image part in the request.files")
        return jsonify({'error': 'No image provided'})
    
    image = request.files['image']
    if image.filename == '':
        print("No image selected for uploading")
        return jsonify({'error': 'No image selected for uploading'})

        img = soil_image(Image.open(image))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        
        print(f"Prediction: {prediction}")

        predicted_class = class_names[np.argmax(prediction)]
        print(f"Predicted class: {predicted_class}")
        
        return render_template('result.html', result=predicted_class)


if __name__ == "__main__":
    app.run(debug=True)
