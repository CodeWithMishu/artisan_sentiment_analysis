#!/usr/bin/env bash
set -e
# Run HP Artisan Intelligence (Unix)
# Create virtual environment if missing
python3 -m venv .venv || python -m venv .venv || true
# Activate
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

pip install -r requirements.txt || true

# Generate simulated data
python data_simulator.py || { echo "Data simulation failed"; exit 1; }

# Optional: setup MongoDB (may require local mongod running)
python scripts/setup_database.py || echo "Database setup skipped or failed"

# Run analysis
python scripts/run_analysis.py --mode simulation || { echo "Analysis failed"; exit 1; }

# Start dashboard in background and log output
nohup python government_dashboard.py > logs/dashboard.log 2>&1 &

echo "Dashboard started (logs/dashboard.log)"
