"""
Register existing trained model to MLflow
Run this ONCE to register your already-trained model
"""

import mlflow
import mlflow.sklearn
import joblib
import yaml
from datetime import datetime

def register_existing_model():
    """Register your existing model to MLflow"""
    
    # Load config
    with open('../config/mlflow_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    
    print("\n" + "="*70)
    print("📦 Registering Existing Model to MLflow")
    print("="*70)
    
    # Load your existing model
    model_path = '../backend/models/best_eladfp_model.pkl'
    print(f"\n📂 Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        return
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"existing_model_registration_{datetime.now().strftime('%Y%m%d')}"):
        
        # Set tags
        print("\n🏷️ Setting tags...")
        for key, value in config['tags'].items():
            mlflow.set_tag(key, value)
        
        mlflow.set_tag("registration_date", datetime.now().isoformat())
        mlflow.set_tag("model_source", "pre-trained")
        
        # Log model metadata (without actual metrics since we're not retraining)
        print("\n📊 Logging model metadata...")
        
        # If you know your model's parameters, log them
        mlflow.log_param("model_type", "VotingClassifier")
        mlflow.log_param("estimators", "RandomForest, LogisticRegression, XGBoost")
        mlflow.log_param("voting", "soft")
        
        # Expected features
        expected_features = [
            'profile pic', 'nums/length username', 'fullname words', 
            'nums/length fullname', 'name==username', 'description length',
            'external URL', 'private', 'log_posts', 'log_followers', 
            'log_following', 'followers_to_following', 'posts_per_follower'
        ]
        mlflow.log_param("input_features", expected_features)
        mlflow.log_param("n_features", len(expected_features))
        
        # If you have test metrics from your original training, add them here
        # mlflow.log_metric("test_accuracy", 0.95)  # Example
        # mlflow.log_metric("test_f1", 0.93)         # Example
        
        # Log the model
        print("\n🤖 Registering model...")
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=config['model_name'],
            signature=None  # We'll skip signature for now
        )
        
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "="*70)
        print(f"✅ Model Registered Successfully!")
        print(f"Run ID: {run_id}")
        print(f"Model Name: {config['model_name']}")
        print("="*70)
        
        print("\n💡 Next steps:")
        print("1. Run: mlflow ui")
        print("2. Open: http://localhost:5000")
        print("3. View your registered model in the Models tab")
        
        return run_id

if __name__ == '__main__':
    # Install pyyaml if not installed
    try:
        import yaml
    except ImportError:
        print("Installing pyyaml...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyyaml'])
        import yaml
    
    register_existing_model()