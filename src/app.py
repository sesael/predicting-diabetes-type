from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('models/final_xgboost_model.pkl')

# Route for the homepage (input form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    insulin_levels = float(request.form['Insulin_Levels'])
    age = float(request.form['Age'])
    bmi = float(request.form['BMI'])
    blood_pressure = float(request.form['Blood_Pressure'])
    cholesterol_levels = float(request.form['Cholesterol_Levels'])
    waist_circumference = float(request.form['Waist_Circumference'])
    blood_glucose_levels = float(request.form['Blood_Glucose_Levels'])
    weight_gain_pregnancy = float(request.form['Weight_Gain_Pregnancy'])
    pancreatic_health = float(request.form['Pancreatic_Health'])
    pulmonary_function = float(request.form['Pulmonary_Function'])
    neurological_assessments = float(request.form['Neurological_Assessments'])
    digestive_enzyme_levels = float(request.form['Digestive_Enzyme_Levels'])
    birth_weight = float(request.form['Birth_Weight'])

    # Prepare the data for prediction (ensure the features are in the correct order)
    data = np.array([[insulin_levels, age, bmi, blood_pressure, cholesterol_levels, waist_circumference, 
                      blood_glucose_levels, weight_gain_pregnancy, pancreatic_health, pulmonary_function, 
                      neurological_assessments, digestive_enzyme_levels, birth_weight]])
    
    # Make the prediction using the loaded model
    prediction = model.predict(pd.DataFrame(data, columns=[
        'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels', 
        'Waist Circumference', 'Blood Glucose Levels', 'Weight Gain During Pregnancy', 
        'Pancreatic Health', 'Pulmonary Function', 'Neurological Assessments', 
        'Digestive Enzyme Levels', 'Birth Weight'
    ]))

    # Mapping the prediction (this assumes the model returns an index that maps to the type of diabetes)
    diabetes_types = {
        0: 'Steroid-Induced Diabetes', 
        1: 'Neonatal Diabetes Mellitus (NDM)', 
        2: 'Prediabetic', 
        3: 'Type 1 Diabetes', 
        4: 'Wolfram Syndrome', 
        5: 'LADA', 
        6: 'Type 2 Diabetes', 
        7: 'Wolcott-Rallison Syndrome', 
        8: 'Secondary Diabetes', 
        9: 'Type 3c Diabetes (Pancreatogenic Diabetes)', 
        10: 'Gestational Diabetes', 
        11: 'Cystic Fibrosis-Related Diabetes (CFRD)', 
        12: 'MODY'
    }

    # Get the predicted diabetes type
    predicted_type = diabetes_types[prediction[0]]

    # Render the result page
    return render_template('result.html', prediction=predicted_type)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)