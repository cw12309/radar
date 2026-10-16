@echo off
echo ========================================
echo RADAR Web Platform v3.0
echo Full-Featured Version
echo ========================================
echo.
echo Features:
echo - Phase 1: Basic diagnosis
echo - Phase 2: 3D visualization
echo - Phase 3: PDF reports, Batch processing, History, Comparison
echo.
call conda activate radar
cd radar-web
echo Starting Streamlit server on port 8503...
echo.
python -m streamlit run app_v3.py --server.headless true --server.port 8503
