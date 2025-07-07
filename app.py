from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("fatigue_neuron_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        motion = int(request.json['motion'])
        temp = int(request.json['temp'])
        
       
        input_data = np.array([[motion, temp]])
        prediction = model.predict(input_data)[0][0]
        fired = int(prediction >= 0.5)

        return jsonify({
            "prediction": fired,
            "message": "✅ Neuron Fired" if fired else "❌ Noise Detected"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
