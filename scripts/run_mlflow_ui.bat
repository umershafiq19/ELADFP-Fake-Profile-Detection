@echo off
echo =======================================
echo   Starting MLflow Dashboard
echo =======================================
echo.
echo Open your browser at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
cd ..
mlflow ui --backend-store-uri ./mlruns --port 5000
pause