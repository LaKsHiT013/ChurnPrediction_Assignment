from flask import Flask, request, jsonify, render_template, send_file
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Selected features for the model
selected_features = ['MonthlyCharges', 'tenure', 'TotalCharges', 
                     'PaymentMethod_Electronic check', 
                     'InternetService_Fiber optic', 
                     'TechSupport_Yes', 
                     'PaperlessBilling', 
                     'OnlineSecurity_Yes', 
                     'SeniorCitizen', 
                     'StreamingBoth']

# Route for the homepage with a form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting form data and converting to a pandas DataFrame
        data = request.form
        
        # Prepare input data based on the selected features
        input_data = {
            'MonthlyCharges': [float(data.get('MonthlyCharges'))],
            'tenure': [float(data.get('tenure'))],
            'TotalCharges': [float(data.get('TotalCharges'))],
            'PaymentMethod_Electronic check': [1 if data.get('PaymentMethod') == 'Electronic check' else 0],
            'InternetService_Fiber optic': [1 if data.get('InternetService') == 'Fiber optic' else 0],
            'TechSupport_Yes': [1 if data.get('TechSupport') == 'Yes' else 0],
            'PaperlessBilling': [1 if data.get('PaperlessBilling') == 'Yes' else 0],
            'OnlineSecurity_Yes': [1 if data.get('OnlineSecurity') == 'Yes' else 0],
            'SeniorCitizen': [int(data.get('SeniorCitizen'))],
            'StreamingBoth': [1 if data.get('StreamingBoth') == 'Yes' else 0]
        }
        
        # Creating a DataFrame from the input data
        df = pd.DataFrame(input_data)
        
        # Apply the scaler to the appropriate features
        df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        
        # Model prediction
        prediction = model.predict(df)
        
        # Sending prediction result to the front-end
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

# Route to serve the Pandas Profiling report
@app.route('/report')
def report():
    # Use os.path.join to ensure correct file path resolution
    file_path = os.path.join(os.getcwd(), 'Analysis.html')
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True)