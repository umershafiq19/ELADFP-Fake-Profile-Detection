import mlflow
import mlflow.sklearn
from datetime import datetime
import os
import re

class PredictionLogger:
    """Log predictions to MLflow for monitoring"""
    
    def __init__(self, tracking_uri=None, experiment_name='ELADFP-Predictions'):
        if tracking_uri is None:
        # Get absolute path to mlruns directory
            import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        tracking_uri = os.path.join(base_dir, 'mlruns')
    
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.enabled = True
        print(f"✅ MLflow prediction logging enabled (URI: {tracking_uri})")
    except Exception as e:
        print(f"⚠️ MLflow logging disabled: {e}")
        self.enabled = False
    
    def sanitize_param_name(self, name):
        """Sanitize parameter name for MLflow (remove special chars)"""
        # Replace == with _equals_
        name = name.replace('==', '_equals_')
        # Replace # with hash_
        name = name.replace('#', 'hash_')
        # Replace / with _per_
        name = name.replace('/', '_per_')
        # Remove any other invalid characters
        name = re.sub(r'[^a-zA-Z0-9_\-\.\s]', '_', name)
        return name
    
    def log_prediction(self, input_features, prediction, confidence, reasoning=None):
        """Log a single prediction"""
        if not self.enabled:
            return
        
        try:
            with mlflow.start_run(run_name=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log input features as parameters (with sanitized names)
                for feature, value in input_features.items():
                    if isinstance(value, (int, float, str)):
                        sanitized_name = self.sanitize_param_name(feature)
                        mlflow.log_param(f"input_{sanitized_name}", value)
                
                # Log prediction results as metrics
                mlflow.log_metric("predicted_fake", int(prediction))
                mlflow.log_metric("confidence_real", confidence['real_profile_prob'])
                mlflow.log_metric("confidence_fake", confidence['fake_profile_prob'])
                
                # Log tags
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                mlflow.set_tag("prediction_type", "FAKE" if prediction == 1 else "REAL")
                
                # Log reasoning as artifact (optional)
                if reasoning:
                    reasoning_file = f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(reasoning_file, 'w', encoding='utf-8') as f:
                        f.write(reasoning)
                    mlflow.log_artifact(reasoning_file)
                    os.remove(reasoning_file)
        
        except Exception as e:
            print(f"⚠️ MLflow logging error: {e}")