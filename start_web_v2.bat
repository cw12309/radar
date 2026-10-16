@echo off
echo ========================================
echo RADAR Web Platform v2.0
echo Phase 2 Enhanced Version
echo ========================================
echo.
call conda activate radar
cd radar-web
echo Starting Streamlit server...
echo.
python -m streamlit run app_v2.py --server.headless true --server.port 8502
