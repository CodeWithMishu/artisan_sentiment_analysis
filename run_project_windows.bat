@echo off
REM Run HP Artisan Intelligence (Windows)

:: Create virtual environment if missing
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt

:: Generate simulated data
python data_simulator.py

:: Optional: setup MongoDB database (if running MongoDB)
python scripts\setup_database.py || echo Database setup skipped or failed

:: Run analysis (simulation mode)
python scripts\run_analysis.py --mode simulation

:: Start dashboard in a new window
start cmd /k python government_dashboard.py

:: Pause to keep window open
pause
