# app.py
# Titanic Survival Prediction - Flask Web Application

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and preprocessors
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')
le_sex = joblib.load('model/le_sex.pkl')
le_embarked = joblib.load('model/le_embarked.pkl')

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Encode categorical variables
        sex_encoded = le_sex.transform([sex])[0]
        embarked_encoded = le_embarked.transform([embarked])[0]
        
        # Prepare features array
        features = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': 'Survived' if prediction == 1 else 'Did Not Survive',
            'survival_probability': f"{probability[1]*100:.2f}%",
            'death_probability': f"{probability[0]*100:.2f}%",
            'confidence': f"{max(probability)*100:.2f}%",
            'class': prediction
        }
        
        return render_template('index.html', result=result, form_data=request.form)
    
    except Exception as e:
        error = f"Error: {str(e)}"
        return render_template('index.html', error=error)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Extract features
        pclass = int(data['pclass'])
        sex = data['sex']
        age = float(data['age'])
        fare = float(data['fare'])
        embarked = data['embarked']
        
        # Encode categorical variables
        sex_encoded = le_sex.transform([sex])[0]
        embarked_encoded = le_embarked.transform([embarked])[0]
        
        # Prepare features array
        features = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Return JSON response
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'result': 'Survived' if prediction == 1 else 'Did Not Survive',
            'probabilities': {
                'survived': float(probability[1]),
                'not_survived': float(probability[0])
            },
            'confidence': float(max(probability))
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Check if model files exist
    required_files = [
        'model/titanic_survival_model.pkl',
        'model/scaler.pkl',
        'model/le_sex.pkl',
        'model/le_embarked.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: Missing model files!")
        print("Please run model_building.py first to generate the model files.")
        print(f"Missing files: {missing_files}")
    else:
        print("=" * 60)
        print("TITANIC SURVIVAL PREDICTION - WEB APPLICATION")
        print("=" * 60)
        print("\nServer starting...")
        print("Open your browser and navigate to: http://localhost:5000")
        print("\nPress CTRL+C to stop the server")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)