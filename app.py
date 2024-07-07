from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import pickle

app = Flask(__name__)
model = load_model('models/soil_model.h5')
model_nitro = pickle.load(open('models/model_nitro.pkl', 'rb'))
model_phos = pickle.load(open('models/model_phos.pkl', 'rb'))
model_pot = pickle.load(open('models/model_pot.pkl', 'rb'))
model_fert = pickle.load(open('models/model_fert.pkl', 'rb'))

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
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    img = soil_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    try:
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
    except ValueError:
        return jsonify({'error': 'Invalid rainfall or temperature'}), 400

    prediction_value = prediction.flatten()[0]

    nitro = model_nitro.predict([[prediction_value, rainfall, temperature]])[0]
    phos = model_phos.predict([[prediction_value, rainfall, temperature]])[0]
    pot = model_pot.predict([[prediction_value, rainfall, temperature]])[0]
    fert = model_fert.predict([[prediction_value, rainfall, temperature]])[0]

    predicted_class = class_names[np.argmax(prediction)]
    
    return render_template('result.html', result=predicted_class, nitro=nitro.tolist(), phos=phos.tolist(), pot=pot.tolist(), fert=fert.tolist())


if __name__ == "__main__":
    app.run(debug=True)
