import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import yaml
import os
from datetime import datetime

class MLflowTrainer:
    """Train models with MLflow tracking"""
    
    def __init__(self, config_path='../config/mlflow_config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.config['tracking_uri'])
        
        # Set experiment
        mlflow.set_experiment(self.config['experiment_name'])
        
        print(f"✅ MLflow configured: Experiment '{self.config['experiment_name']}'")
    
    def prepare_data(self, data_path):
        """Load and prepare data"""
        print(f"📂 Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('fake', axis=1)
        y = df['fake']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_ensemble_model(self, params=None):
        """Create ensemble model with Random Forest, Logistic Regression, XGBoost"""
        if params is None:
            params = {
                'rf_n_estimators': 100,
                'rf_max_depth': 10,
                'lr_C': 1.0,
                'xgb_n_estimators': 100,
                'xgb_max_depth': 5,
                'xgb_learning_rate': 0.1
            }
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            random_state=42
        )
        
        # Logistic Regression
        lr = LogisticRegression(
            C=params['lr_C'],
            max_iter=1000,
            random_state=42
        )
        
        # XGBoost
        xgb = XGBClassifier(
            n_estimators=params['xgb_n_estimators'],
            max_depth=params['xgb_max_depth'],
            learning_rate=params['xgb_learning_rate'],
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Voting Classifier (Ensemble)
        ensemble = VotingClassifier(
            estimators=[
                ('random_forest', rf),
                ('logistic_regression', lr),
                ('xgboost', xgb)
            ],
            voting='soft'
        )
        
        return ensemble, params
    
    def train_and_log(self, X_train, X_test, y_train, y_test, params=None, run_name=None):
        """Train model and log everything to MLflow"""
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            print("\n" + "="*70)
            print("🚀 Starting MLflow Training Run")
            print("="*70)
            
            # Log tags
            for key, value in self.config['tags'].items():
                mlflow.set_tag(key, value)
            
            # Create model
            model, model_params = self.create_ensemble_model(params)
            
            # Log parameters
            print("\n📊 Logging parameters...")
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)
            
            mlflow.log_param('train_samples', len(X_train))
            mlflow.log_param('test_samples', len(X_test))
            mlflow.log_param('features', list(X_train.columns))
            
            # Train model
            print("\n🔧 Training model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            print("\n📈 Calculating metrics...")
            
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'test_precision': precision_score(y_test, y_pred_test),
                'test_recall': recall_score(y_test, y_pred_test),
                'test_f1': f1_score(y_test, y_pred_test),
                'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            metrics['cv_mean_accuracy'] = cv_scores.mean()
            metrics['cv_std_accuracy'] = cv_scores.std()
            
            # Log metrics
            print("\n📊 Logging metrics...")
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                print(f"  {metric_name}: {metric_value:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            tn, fp, fn, tp = cm.ravel()
            
            mlflow.log_metric('true_negatives', int(tn))
            mlflow.log_metric('false_positives', int(fp))
            mlflow.log_metric('false_negatives', int(fn))
            mlflow.log_metric('true_positives', int(tp))
            
            # Classification report
            report = classification_report(y_test, y_pred_test, output_dict=True)
            
            # Log artifacts
            print("\n💾 Logging artifacts...")
            
            # Save classification report
            report_path = 'classification_report.txt'
            with open(report_path, 'w') as f:
                f.write(classification_report(y_test, y_pred_test))
            mlflow.log_artifact(report_path)
            os.remove(report_path)
            
            # Save confusion matrix
            cm_path = 'confusion_matrix.txt'
            with open(cm_path, 'w') as f:
                f.write(f"Confusion Matrix:\n{cm}\n\n")
                f.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            
            # Log model
            print("\n🤖 Logging model...")
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=self.config['model_name']
            )
            
            # Save model locally
            model_path = '../backend/models/best_eladfp_model.pkl'
            joblib.dump(model, model_path)
            print(f"✅ Model saved to {model_path}")
            
            # Get run info
            run_id = mlflow.active_run().info.run_id
            
            print("\n" + "="*70)
            print(f"✅ MLflow Run Completed: {run_id}")
            print("="*70)
            
            return model, metrics, run_id


def main():
    """Main training function"""
    
    # Initialize trainer
    trainer = MLflowTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        'data/processed/train_test_split.csv'
    )
    
    # Train with default parameters
    print("\n🎯 Training with default parameters...")
    model, metrics, run_id = trainer.train_and_log(
        X_train, X_test, y_train, y_test,
        run_name="ensemble_baseline"
    )
    
    print("\n✅ Training Complete!")
    print(f"Run ID: {run_id}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")
    
    # Optional: Train with different hyperparameters
    print("\n🎯 Training with tuned parameters...")
    tuned_params = {
        'rf_n_estimators': 150,
        'rf_max_depth': 15,
        'lr_C': 0.5,
        'xgb_n_estimators': 120,
        'xgb_max_depth': 6,
        'xgb_learning_rate': 0.05
    }
    
    model_tuned, metrics_tuned, run_id_tuned = trainer.train_and_log(
        X_train, X_test, y_train, y_test,
        params=tuned_params,
        run_name="ensemble_tuned"
    )
    
    print("\n✅ Tuned Training Complete!")
    print(f"Run ID: {run_id_tuned}")
    print(f"Test Accuracy: {metrics_tuned['test_accuracy']:.4f}")
    print(f"Test F1 Score: {metrics_tuned['test_f1']:.4f}")


if __name__ == '__main__':
    main()