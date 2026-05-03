#!/bin/bash

# Quantra setup script

echo "Setting up Quantra..."

# Directory structure
mkdir -p src/backend/models
mkdir -p src/backend/data
mkdir -p src/backend/utils
mkdir -p src/backend/backtesting/data
mkdir -p src/backend/results

# Package markers
touch src/backend/__init__.py
touch src/backend/models/__init__.py
touch src/backend/data/__init__.py
touch src/backend/utils/__init__.py
touch src/backend/backtesting/__init__.py

# Python deps
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Sample data check
if [ ! -f "src/backend/data/all_options_all_dates.csv" ]; then
    echo "Sample SPX options data not found."
    echo "Add your dataset to src/backend/data/all_options_all_dates.csv"
    echo "Expected columns: contractSymbol, strike, bid, ask, impliedVolatility, Expiration, Type, etc."
fi

# Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/backend"

# Verify install
python -c "
import numpy, pandas, scipy, sklearn, yfinance, plotly, streamlit
print('Core packages OK')

try:
    import sys, os
    sys.path.append('src/backend')
    from models.option_models import OptionPricingModels
    from data.data_fetcher import DataFetcher
    print('Custom modules OK')
except Exception as e:
    print(f'Module import warning: {e}')
"

echo ""
echo "Setup complete."
echo ""
echo "Run the app:"
echo "    streamlit run src/streamlit.py"
echo ""
echo "Backtests write to ./src/backend/results"
