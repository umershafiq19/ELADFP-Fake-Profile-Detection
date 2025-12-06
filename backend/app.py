from flask import Flask, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from backend.routes.predict import predict_bp
from backend.routes.batch import batch_bp
from backend.routes.health import health_bp

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)
    
    # Enable CORS with specific configuration
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Endpoint not found', 'message': str(e)}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        return jsonify({'error': 'Unexpected error', 'message': str(e)}), 500
    
    # Register blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(health_bp)
    
    # Root endpoint for testing
    @app.route('/')
    def home():
        return jsonify({
            'message': '🚀 ELADFP API is running',
            'endpoints': {
                '/health': 'Health check',
                '/predict': 'Single profile prediction (POST)',
                '/batch': 'Batch predictions (POST)'
            }
        }), 200
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    print("\n" + "="*70)
    print("🚀 ELADFP Flask API Starting...")
    print(f"📍 Host: {host}:{port}")
    print(f"🐛 Debug: {debug}")
    print(f"🔗 API URL: http://localhost:{port}")
    print(f"🏥 Health Check: http://localhost:{port}/health")
    print("="*70 + "\n")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")