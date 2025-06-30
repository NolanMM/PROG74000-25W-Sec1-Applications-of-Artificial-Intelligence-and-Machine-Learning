from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from imblearn.combine import SMOTEENN
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

def initialize_model():
    """Load and train the model once during startup"""
    global model, scaler, feature_names
    
    # Load training data
    train_data = pd.read_csv('./Dataset/cleaned_customer_churn_dataset_training.csv')
    
    # Prepare features and target
    X_train = train_data.drop(['CustomerID', 'Churn'], axis=1)
    y_train = train_data['Churn']
    feature_names = X_train.columns
    print("Feature names:")
    print(feature_names)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Handle class imbalance
    smote_enn = SMOTEENN(sampling_strategy=0.8, random_state=42)
    X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_scaled, y_train)
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=20,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train_balanced, y_train_balanced)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict customer churn"""
    try:
        # Get input data from request
        data = request.get_json()
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {missing_features}'
            }), 400
            
        # Order features correctly and scale
        X_input = input_df[feature_names]
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = int(probability >= 0.5)
        
        return jsonify({
            'prediction': prediction,
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, host='127.0.0.1', port=5000)