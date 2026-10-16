@echo off
call conda activate radar
cd radar-web
python -m streamlit run app.py --server.headless true --server.port 8501
