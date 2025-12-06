from flask import Blueprint, jsonify
import joblib
import os

health_bp = Blueprint('health', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_eladfp_model.pkl')

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Check API and model status"""
    model_loaded = os.path.exists(MODEL_PATH)
    
    return jsonify({
        'status': 'ELADFP Flask API Running',
        'model_loaded': model_loaded,
        'timestamp': str(pd.Timestamp.now())
    }), 200

@health_bp.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'project': 'ELADFP: Ensemble Learning Fake Profile Detection',
        'team': ['Fattah Ali 21L-5187', 'Ubaid Ur Rehman 21L-5189', 'Umer Shafiq 21L-5401'],
        'supervisor': 'Rana Waqas Ali',
        'endpoints': {
            'POST /predict': 'Predict single profile',
            'POST /batch_predict': 'Batch predict from JSON file',
            'GET /health': 'Check API status',
            'GET /': 'API documentation'
        },
        'expected_input_fields': [
            'profile pic', 'username', 'fullname', 'bio',
            'external URL', 'private', '#posts', '#followers', '#following'
        ]
    }), 200