from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os
from backend.preprocessing.feature_calculator import FeatureCalculator
from backend.preprocessing.validators import InputValidator
from backend.ai_reasoning.gemini_analyzer import GeminiAnalyzer
from backend.routes.mlflow_logging import PredictionLogger

predict_bp = Blueprint('predict_bp', __name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_eladfp_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}")
    model = None

gemini = GeminiAnalyzer()

# Initialize prediction logger (moved after model loading)
try:
    prediction_logger = PredictionLogger()
    print("✅ MLflow prediction logging enabled")
except Exception as e:
    prediction_logger = None
    print(f"⚠️ MLflow prediction logging disabled: {e}")

MODEL_FEATURES = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private',
    'log_posts', 'log_followers', 'log_following', 'followers_to_following', 'posts_per_follower'
]

@predict_bp.route('/predict', methods=['POST'])
def predict_single():
    """Predict if a single profile is fake or real"""
    try:
        data = request.get_json()
        print("\n" + "="*70)
        print("📩 Received data for prediction:")
        print(data)
        print("="*70)
        
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Validate input
        is_valid, errors = InputValidator.validate_input(data)
        if not is_valid:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Calculate features
        processed_data = FeatureCalculator.preprocess_features(data)
        
        # Convert to DataFrame
        df = pd.DataFrame([processed_data])
        
        # Apply log transforms
        df = FeatureCalculator.apply_log_transforms(df)
        
        # Ensure correct feature order
        df = df[MODEL_FEATURES]
        
        print("\n🧹 Preprocessed input:")
        print(df.to_string(index=False))
        
        # Make prediction
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        real_prob, fake_prob = float(proba[0]), float(proba[1])
        
        print(f"✅ Prediction → {'FAKE' if prediction == 1 else 'REAL'} | Fake Prob = {fake_prob:.3f}")
        
        # Generate reasoning
        confidence = {'real_profile_prob': real_prob, 'fake_profile_prob': fake_prob}
        reasoning = gemini.generate_reasoning(processed_data, prediction, confidence)
        
        response = {
            'prediction': {'is_fake': int(prediction)},
            'confidence': confidence,
            'reasoning': reasoning,
            'message': '⚠️ FAKE PROFILE DETECTED!' if prediction == 1 else '✅ Real Profile',
            'features_used': processed_data
        }
        
        print("\n📤 Response Sent:")
        print(response)
        print("="*70)
        
        # Log to MLflow (MOVED INSIDE try block, BEFORE return)
        if prediction_logger:
            try:
                prediction_logger.log_prediction(processed_data, prediction, confidence, reasoning)
                print("✅ Prediction logged to MLflow")
            except Exception as e:
                print(f"⚠️ MLflow logging error: {e}")
        
        return jsonify(response), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500