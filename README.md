# MECE 689: DRL vs Forecast-Based Inventory Policies

Single-product, single-warehouse lost-sales simulator (T=90, lead time L=2).

## Run
pip install -r requirements.txt
pytest -v
python run_baselines.py

## Status
Version 0 (stationary Poisson demand): simulator, Gymnasium env, baselines done.