#!/bin/bash

# Advanced Options Pricing Platform Setup Script

echo "🚀 Setting up Advanced Options Pricing Platform..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p src/backend/models
mkdir -p src/backend/data  
mkdir -p src/backend/utils
mkdir -p src/backend/backtesting/data
mkdir -p src/frontend
mkdir -p output

# Create __init__.py files for Python packages
touch src/backend/__init__.py
touch src/backend/models/__init__.py
touch src/backend/data/__init__.py
touch src/backend/utils/__init__.py
touch src/backend/backtesting/__init__.py

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download sample data (if not exists)
echo "📊 Checking for sample data..."
if [ ! -f "src/backend/data/all_options_all_dates.csv" ]; then
    echo "⚠️  Sample SPX options data not found."
    echo "   Please add your options dataset to src/backend/data/all_options_all_dates.csv"
    echo "   Expected columns: contractSymbol, strike, bid, ask, impliedVolatility, Expiration, Type, etc."
fi

# Set Python path
echo "🐍 Setting up Python path..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/backend"

# Create output directory
echo "📈 Setting up output directory..."
mkdir -p output/backtesting
mkdir -p output/iv_surfaces
mkdir -p output/reports

# Verify installation
echo "✅ Verifying installation..."
python -c "
import numpy, pandas, scipy, sklearn, yfinance, plotly, streamlit
print('✅ All core packages imported successfully')

try:
    import sys
    sys.path.append('src/backend')
    from models.option_models import OptionPricingModels
    from data.data_fetcher import DataFetcher
    print('✅ Custom modules loaded successfully')
except Exception as e:
    print(f'⚠️  Module import warning: {e}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "   1. Run the Streamlit app: streamlit run src/frontend/streamlit.py"
echo "   2. Or run backtesting: python -c 'import sys; sys.path.append(\"src/backend\"); from backtesting.backtester import SPXBacktester; bt = SPXBacktester(); bt.run_full_analysis()'"
echo "   3. Check output folder for generated reports and visualizations"
echo ""
echo "📚 For detailed usage instructions, see README.md"
echo ""
