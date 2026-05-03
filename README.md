# Quantra: Real-time options pricing & implied volatility fititng

Quantra is an online, educational options pricing and analysis platform that provides real-time low-latency pricing, backtesting capabilities, and professional-grade volatility fitting through an intuitive Streamlit web interface. Quantra was developed to provide MATH 86 students at Dartmouth College with a platform to understand firsthand relationships & sesnitivities among the Greeks, hedging & portfolio management workflows, and how traditional pricing models are deployed.

[**Website Link**](https://advancedoptionspricing.streamlit.app/)

<img width="1119" height="729" alt="Screenshot 2026-05-03 at 5 19 28 PM" src="https://github.com/user-attachments/assets/c842c3f6-204f-4117-809e-ce0855d2a3d3" />

## Pricing Infrastructure

<img width="1114" height="857" alt="Screenshot 2026-05-03 at 5 20 47 PM" src="https://github.com/user-attachments/assets/e6a0460e-5bf9-4358-992e-ebf451f732d1" />

<img width="1049" height="751" alt="Screenshot 2026-05-03 at 5 27 15 PM" src="https://github.com/user-attachments/assets/3115e85f-a0b4-428c-a7db-3f23d5da9f49" />


<img width="1114" height="857" alt="Screenshot 2026-05-03 at 5 20 47 PM" src="https://github.com/user-attachments/assets/c3371128-b064-40dc-b45d-6bbbc9b54860" />


### Backtesting
- Pricing models are calibrated via a comprehensive historical backtesting engine using SPX options data
- Data is pre-processed and filtered to only yield high-quality contracts, and predicted vs. actual prices are measured
- Models are calibrated to SPX surfaces & optimized for backtesting




## Features


### 1. Option Pricing Calculator

   - Parameters: Ticker symbol, option type (Call/Put), strike price and expiration, risk-free rate and volatility (Use Live Market Data/Historical)
   - Automatic price and volatility fetching; real-time risk-free rate updates
   - All model prices display side-by-side with complete Greeks calculation
   - Price sensitivity analysis & Monte Carlo path visualization
- Performance Metrics:
    - MAE, RMSE, percentage errors across all models
    - Statistical significance testing
    - Error distribution analysis
    - Performance by option characteristics


** Models **
1. Black-Scholes Model:
    - Analytical European option pricing with complete Greeks
    - See detailed literature & research notes in `literature.md`
    - Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637-654
2. Monte Carlo Simulation:
    - Heston stochastic volatility model with multi-path visualization  
3. Binomial Tree Method:
     - Discrete-time lattice approach for American/European options
       
   - Parameters: Ticker symbol, option type (Call/Put), strike price and expiration, risk-free rate and volatility (Use Live Market Data/Historical)
   - Automatic price and volatility fetching; real-time risk-free rate updates
   - All model prices display side-by-side with complete Greeks calculation
   - Price sensitivity analysis & Monte Carlo path visualization
- Performance Metrics:
    - MAE, RMSE, percentage errors across all models
    - Statistical significance testing
    - Error distribution analysis
    - Performance by option characteristics
- Interactive Visualizations:
    - 3D heatmaps & convergence analysis for pricing models
    - Model accuracy scatter plots & error distribution box plots for backtesting
    - 3D volatility surfaces across strikes and expirations for volatility smile analysis

- <img width="740" alt="Screenshot 2025-06-29 at 10 14 27 PM" src="https://github.com/user-attachments/assets/fb9d24a3-b063-4c2e-b18b-cdb4f978f6ac" />

## Usage

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

```bash
# Clone the repository
git clone [your-repository-url]
cd quant-options-pricing

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Quick Start

#### 1. Launch Streamlit Application

```bash
# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"

# Run the main application
streamlit run frontend/streamlit.py
```

The application will open in your browser at `http://localhost:8501`

#### 2. Run Backtesting Analysis

```bash
# Jupyter notebook
jupyter notebook backend/backtesting/backtesting.ipynb
```

#### 3. Data Requirements

Use the default SPX options dataset or use your SPX options dataset in `src/backend/data/all_options_all_dates.csv`

**Expected CSV columns:**
- `contractSymbol`: Option contract identifier
- `strike`: Strike price
- `bid`: Bid price
- `ask`: Ask price
- `impliedVolatility`: Market implied volatility
- `Expiration`: Expiration date
- `Type`: 'call' or 'put'
- `lastTradeDate`: Trade date
- `volume`: Trading volume
- `openInterest`: Open interest

## Advanced Configurations

### Custom Risk-Free Rate
```python
# In your Python scripts
from backend.data.data_fetcher import DataFetcher
fetcher = DataFetcher()
fetcher.risk_free_rate = 0.045  # 4.5%
```

### Monte Carlo Parameters
```python
# Adjust simulation parameters
from backend.models.option_models import OptionPricingModels
pricing = OptionPricingModels(S, K, T, r, sigma, option_type)
mc_price, paths = pricing.new_monte_carlo_option_price(num_simulations=50000)
```

### Binomial Tree Resolution
```python
# Higher resolution binomial tree
bt_price = pricing.binomial_tree_option_price(N=200)  # 200 steps
```

## License

MIT License - Open source for educational and research purposes.
