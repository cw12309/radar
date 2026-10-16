@echo off
echo ========================================
echo RADAR Medical Imaging Platform
echo Flask Professional Edition v4.0
echo ========================================
echo.

cd /d %~dp0

echo Activating conda environment: radar
call conda activate radar

echo Installing Flask dependencies...
pip install -q flask flask-cors

echo Starting Flask server on port 5000...
echo.
echo Access the application at:
echo    Local:   http://localhost:5000
echo    Network: http://YOUR_IP:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py

pause
