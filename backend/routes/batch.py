from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os
import json
from backend.preprocessing.feature_calculator import FeatureCalculator
from backend.preprocessing.validators import InputValidator
from backend.ai_reasoning.gemini_analyzer import GeminiAnalyzer

batch_bp = Blueprint('batch', __name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_eladfp_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

gemini = GeminiAnalyzer()

MODEL_FEATURES = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private',
    'log_posts', 'log_followers', 'log_following', 'followers_to_following', 'posts_per_follower'
]

@batch_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple profiles from JSON file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'File must be JSON format'}), 400
        
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Read JSON file
        data = json.load(file)
        
        if not isinstance(data, list):
            data = [data]
        
        results = []
        errors = []
        
        for idx, profile in enumerate(data):
            try:
                # Validate input
                is_valid, validation_errors = InputValidator.validate_input(profile)
                if not is_valid:
                    errors.append({
                        'profile_index': idx,
                        'error': 'Validation failed',
                        'details': validation_errors
                    })
                    continue
                
                # Calculate features
                processed_data = FeatureCalculator.preprocess_features(profile)
                
                # Convert to DataFrame
                df = pd.DataFrame([processed_data])
                
                # Apply log transforms
                df = FeatureCalculator.apply_log_transforms(df)
                
                # Ensure correct feature order
                df = df[MODEL_FEATURES]
                
                # Make prediction
                prediction = model.predict(df)[0]
                proba = model.predict_proba(df)[0]
                real_prob, fake_prob = float(proba[0]), float(proba[1])
                
                confidence = {'real_profile_prob': real_prob, 'fake_profile_prob': fake_prob}
                reasoning = gemini.generate_reasoning(processed_data, prediction, confidence)
                
                results.append({
                    'profile_index': idx,
                    'prediction': {'is_fake': int(prediction)},
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'message': '⚠️ FAKE' if prediction == 1 else '✅ REAL'
                })
            
            except Exception as e:
                errors.append({
                    'profile_index': idx,
                    'error': str(e)
                })
        
        return jsonify({
            'total_processed': len(results),
            'total_errors': len(errors),
            'results': results,
            'errors': errors
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500