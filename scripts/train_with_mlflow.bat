@echo off
echo ======================================
echo   ELADFP Model Training with MLflow
echo ======================================
echo.
cd training
python mlflow_trainer.py
echo.
echo ======================================
echo   Training Complete!
echo   View results: mlflow ui
echo ======================================
pause