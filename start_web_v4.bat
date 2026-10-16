@echo off
echo ========================================
echo RADAR Medical Imaging Platform v4.0
echo Professional Edition
echo ========================================
echo.

cd /d %~dp0
cd radar-web

echo Activating conda environment: radar
call conda activate radar

echo Starting Streamlit server on port 8504...
echo.
echo Access the application at:
echo    Local:   http://localhost:8504
echo    Network: http://YOUR_IP:8504
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app_v4.py --server.port 8504 --server.address 0.0.0.0

pause
